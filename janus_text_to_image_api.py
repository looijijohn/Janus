import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
import logging
import base64
from io import BytesIO
from PIL import Image
import torch
import os
import uuid
from typing import Optional, List
import torchvision
import numpy as np
from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
from diffusers.models import AutoencoderKL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pydantic model for request validation
class TextToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt for image generation")
    negative_prompt: Optional[str] = Field(None, max_length=500, description="Negative prompt to avoid certain features")
    steps: int = Field(30, ge=1, le=100, description="Number of inference steps")
    width: int = Field(384, ge=64, le=1024, description="Image width in pixels")
    height: int = Field(384, ge=64, le=1024, description="Image height in pixels")
    seed: Optional[int] = Field(None, ge=-1, description="Random seed for reproducibility, -1 for random")
    cfg_weight: float = Field(2.0, ge=1.0, le=10.0, description="Classifier-free guidance weight")
    batchsize: int = Field(5, ge=1, le=10, description="Batch size for generation")

# Global model and processor variables
vl_gpt = None
vl_chat_processor = None
vae = None
model_path = "deepseek-ai/Janus-Pro-7B"
vae_path = "stabilityai/sdxl-vae"

# Hardcoded credentials (replace with secure storage in production)
VALID_USERNAME = "admin"
VALID_PASSWORD = "securepassword123"  # Use environment variables or a secret manager

# HTTP Basic Authentication setup
security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Verify username and password provided in HTTP Basic Auth.
    """
    correct_username = VALID_USERNAME
    correct_password = VALID_PASSWORD
    if credentials.username != correct_username or credentials.password != correct_password:
        logger.warning(f"Authentication failed for username: {credentials.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Lifespan event to load model at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vl_gpt, vl_chat_processor, vae
    logger.info("Loading Janus-Pro-7B model and VAE...")
    try:
        # Load tokenizer and processor
        logger.info(f"Loading VLChatProcessor from {model_path}")
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        # Load model
        logger.info(f"Loading MultiModalityCausalLM from {model_path}")
        vl_gpt = MultiModalityCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
        # Load VAE
        logger.info(f"Loading AutoencoderKL from {vae_path}")
        try:
            vae = AutoencoderKL.from_pretrained(vae_path)
            vae = vae.to(torch.bfloat16).cuda().eval()
        except Exception as vae_error:
            logger.error(f"Failed to load VAE from {vae_path}: {str(vae_error)}")
            raise HTTPException(status_code=500, detail=f"VAE loading failed: {str(vae_error)}")
        logger.info("Model and VAE loaded successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to load model or VAE: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model or VAE loading failed: {str(e)}")
    finally:
        logger.info("Shutting down, releasing model resources...")
        vl_gpt = None
        vl_chat_processor = None
        vae = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

@torch.inference_mode()
def generate_image(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    vae: AutoencoderKL,
    prompt: str,
    negative_prompt: Optional[str] = None,
    cfg_weight: float = 2.0,
    num_inference_steps: int = 30,
    batchsize: int = 5,
    width: int = 384,
    height: int = 384,
    seed: Optional[int] = None
) -> Image.Image:
    """
    Generate an image using Janus-Pro-7B based on the provided prompt.
    Adapted from the GitHub example.
    """
    try:
        # Set seed for reproducibility
        if seed is not None and seed != -1:
            torch.manual_seed(seed)

        # Prepare conversation template
        conversation = [{"role": "User", "content": prompt}, {"role": "Assistant", "content": ""}]
        if negative_prompt:
            conversation[0]["negative_prompt"] = negative_prompt
        sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=""
        )
        prompt = sft_format + vl_chat_processor.image_gen_tag

        # Tokenize prompt
        input_ids = vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).cuda()
        tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
        tokens[batchsize:, 1:] = vl_chat_processor.pad_id
        inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
        inputs_embeds = inputs_embeds[:, :-1, :]  # Remove last <bog> token

        # Initialize latent space
        z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()
        dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + (1.0 / num_inference_steps)

        # Run ODE steps
        attention_mask = torch.ones((2 * batchsize, inputs_embeds.shape[1] + 577)).to(mmgpt.device)
        attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
        attention_mask = attention_mask.int()
        past_key_values = None
        for step in range(num_inference_steps):
            z_input = torch.cat([z, z], dim=0)  # For CFG
            t = step / num_inference_steps * 1000.0
            t = torch.tensor([t] * z_input.shape[0]).to(dt).cuda()
            z_enc = mmgpt.vision_gen_enc_model(z_input, t)
            z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
            z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
            z_emb = mmgpt.vision_gen_enc_aligner(z_emb)
            llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

            # LLM forward pass
            outputs = mmgpt.language_model.model(
                inputs_embeds=llm_emb,
                use_cache=True,
                attention_mask=attention_mask,
                past_key_values=past_key_values
            )
            if step == 0:
                past_key_values = []
                for kv_cache in outputs.past_key_values:
                    k, v = kv_cache[0], kv_cache[1]
                    past_key_values.append((k[:, :, :inputs_embeds.shape[1], :], v[:, :, :inputs_embeds.shape[1], :]))
                past_key_values = tuple(past_key_values)
            hidden_states = outputs.last_hidden_state

            # Transform hidden states to velocity
            hidden_states = mmgpt.vision_gen_dec_aligner(mmgpt.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :]))
            hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
            v = mmgpt.vision_gen_dec_model(hidden_states, hs, t_emb)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = cfg_weight * v_cond - (cfg_weight - 1.0) * v_uncond
            z = z + dt * v

        # Decode with VAE
        decoded_image = vae.decode(z / vae.config.scaling_factor).sample
        decoded_image = decoded_image.clip(-1.0, 1.0) * 0.5 + 0.5
        # Convert to PIL Image (select first image from batch)
        img_tensor = decoded_image[0].permute(1, 2, 0).cpu().float().numpy()
        img = Image.fromarray((img_tensor * 255).astype(np.uint8))
        return img

    except Exception as e:
        logger.error(f"Image generation failed: {str(e)}")
        raise

@app.post("/sdapi/v1/txt2img")
async def text_to_image(payload: TextToImageRequest, username: str = Depends(verify_credentials)) -> dict:
    """
    Generate an image from a text prompt using Janus-Pro-7B.
    Returns a base64-encoded PNG image in Stable Diffusion-compatible format.
    Requires HTTP Basic Authentication.
    """
    try:
        logger.info(f"Authenticated request from {username}: prompt='{payload.prompt}', steps={payload.steps}, "
                    f"width={payload.width}, height={payload.height}, cfg_weight={payload.cfg_weight}")

        # Validate model availability
        if vl_gpt is None or vl_chat_processor is None or vae is None:
            logger.error("Model or VAE not loaded")
            raise HTTPException(status_code=503, detail="Model or VAE not initialized")

        # Generate image
        image = generate_image(
            mmgpt=vl_gpt,
            vl_chat_processor=vl_chat_processor,
            vae=vae,
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            cfg_weight=payload.cfg_weight,
            num_inference_steps=payload.steps,
            batchsize=payload.batchsize,
            width=payload.width,
            height=payload.height,
            seed=payload.seed
        )

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"Image generated successfully for {username}")
        return {
            "images": [img_str],
            "parameters": payload.dict(),
            "info": {
                "model": "Janus-Pro-7B",
                "request_id": str(uuid.uuid4())
            }
        }

    except Exception as e:
        logger.error(f"Error generating image for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")

@app.get("/health")
async def health_check(username: str = Depends(verify_credentials)):
    """
    Check the health of the API and model availability.
    Requires HTTP Basic Authentication.
    """
    if vl_gpt is None or vl_chat_processor is None or vae is None:
        raise HTTPException(status_code=503, detail="Model or VAE not initialized")
    return {"status": "healthy", "model": "Janus-Pro-7B"}

@app.get("/authorize")
async def authorize(username: str = Depends(verify_credentials)):
    """
    Verify credentials and return a success message.
    Requires HTTP Basic Authentication.
    """
    logger.info(f"Successful authentication for {username}")
    return {"message": f"Authenticated successfully as {username}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)