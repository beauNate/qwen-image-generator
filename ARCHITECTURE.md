# Architecture Overview

```mermaid
flowchart TB
    subgraph Frontend["Frontend (Port 8080)"]
        SG[simple_generator.py]
        HTML[Embedded HTML/CSS/JS]
        SG --> HTML
    end

    subgraph Backend["ComfyUI Backend (Port 8188)"]
        Main[main.py]
        Server[server.py]
        Exec[execution.py]
        Nodes[nodes.py]
        Main --> Server
        Server --> Exec
        Exec --> Nodes
    end

    subgraph Core["Core Engine"]
        direction TB
        Comfy[comfy/]
        ComfyExec[comfy_execution/]
        ComfyAPI[comfy_api/]
        ComfyExtras[comfy_extras/]
    end

    subgraph Models["Models Directory"]
        CLIP[text_encoders/]
        UNET[unet/]
        VAE[vae/]
        LoRA[loras/]
    end

    subgraph CustomNodes["Custom Nodes"]
        GGUF[ComfyUI-GGUF]
    end

    subgraph Output["Generated Content"]
        Images[output/]
        Input[input/]
    end

    %% Connections
    HTML -->|HTTP API| Server
    Server --> Core
    Core --> Models
    Core --> CustomNodes
    Exec --> Images

    classDef frontend fill:#4a9eff,stroke:#333,color:#fff
    classDef backend fill:#ff6b6b,stroke:#333,color:#fff
    classDef core fill:#51cf66,stroke:#333,color:#fff
    classDef storage fill:#ffd43b,stroke:#333,color:#000

    class SG,HTML frontend
    class Main,Server,Exec,Nodes backend
    class Comfy,ComfyExec,ComfyAPI,ComfyExtras core
    class Images,Input,Models storage
```

## Component Overview

| Component | Purpose |
|-----------|---------|
| `simple_generator.py` | Standalone GUI (6500 lines) - HTTP server + embedded UI |
| `server.py` | ComfyUI WebSocket/HTTP server |
| `execution.py` | Workflow execution engine |
| `nodes.py` | Core node definitions |
| `comfy/` | Inference engine, model loading, samplers |
| `comfy_extras/` | Additional node packs |
| `custom_nodes/` | GGUF quantization support |

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant GUI as simple_generator.py
    participant API as ComfyUI API
    participant Engine as Execution Engine
    participant Models as Model Loaders

    User->>GUI: Enter prompt + settings
    GUI->>API: POST /prompt (workflow JSON)
    API->>Engine: Queue workflow
    Engine->>Models: Load CLIP, UNet, VAE
    Models-->>Engine: Models ready
    Engine->>Engine: Run KSampler
    Engine->>Engine: VAE Decode
    Engine-->>API: Save image
    API-->>GUI: WebSocket: execution complete
    GUI-->>User: Display image
```

## File Structure

```
qwen-image-generator/
├── simple_generator.py    # Main GUI application
├── main.py                # ComfyUI entry point
├── server.py              # HTTP/WebSocket server
├── execution.py           # Workflow executor
├── nodes.py               # Node definitions
├── comfy/                 # Core inference
├── comfy_api/             # API layer
├── comfy_execution/       # Execution management
├── comfy_extras/          # Extra nodes
├── custom_nodes/          # GGUF support
├── models/                # AI models
├── output/                # Generated images
└── input/                 # Reference images
```
