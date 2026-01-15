from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch, trimesh
from io import BytesIO
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model
from shap_e.util.notebooks import decode_latent_mesh

app = Flask(__name__)
CORS(app)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model("text300M", device=device)
diffusion = diffusion_from_config("text300M", use_fp16=False)

@app.post("/generate")
def generate():
    try:
        data = request.json
        prompt = data.get("prompt", "")
        fmt = data.get("format", "stl")
        
        latents = sample_latents(
            batch_size=1,
            model=model,
            diffusion=diffusion,
            guidance_scale=10.0,
            model_kwargs={"text": [prompt]},
            progress=False,
            device=device
        )

        mesh = decode_latent_mesh(latents[0])
        output = BytesIO()

        if fmt == "stl":
            trimesh.Trimesh(vertices=mesh.verts, faces=mesh.faces).export(output, file_type='stl')
            mime = "model/stl"
        else:
            mesh.save(output)
            mime = "model/gltf-binary"

        output.seek(0)
        return send_file(output, mimetype=mime, as_attachment=True, download_name=f"model.{fmt}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})

if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)

