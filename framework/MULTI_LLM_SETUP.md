# Multi-Provider LLM Setup Instructions

## What We're Using

1. **OpenAI GPT** - You have API key ✅
2. **Google Gemini** - You have API key ✅
3. **Llama 3.1** - Free via Ollama (local)
4. **Qwen 2.5** - Free via Ollama (local)
5. **Mistral** - Free via Ollama (local) - Bonus option

**Total Cost:** $0 for Ollama models, minimal for OpenAI/Gemini

---

## Step 1: Install Ollama

### Windows Installation

1. Download Ollama from: https://ollama.com/download/windows
2. Run the installer
3. Ollama will start automatically

### Verify Installation

Open PowerShell and run:
```powershell
ollama --version
```

---

## Step 2: Pull Required Models

Run these commands in PowerShell:

```powershell
# Llama 3.1 (8B parameters, ~4.7GB)
ollama pull llama3.1:8b

# Qwen 2.5 (7B parameters, ~4.4GB)
ollama pull qwen2.5:7b

# Mistral (7B parameters, ~4.1GB) - Optional
ollama pull mistral:7b
```

**Note:** Each model takes 5-10 minutes to download.

---

## Step 3: Test Ollama Models

```powershell
# Test Llama
ollama run llama3.1:8b "Hello, who are you?"

# Test Qwen
ollama run qwen2.5:7b "Hello, who are you?"
```

Press `Ctrl+D` to exit the chat.

---

## Step 4: Update .env File

Add your OpenAI key to `.env`:

```bash
# Google Gemini (already have)
GEMINI_API_KEY=your_gemini_key_here

# OpenAI GPT (add this)
OPENAI_API_KEY=your_openai_key_here

# Ollama (local, no key needed)
OLLAMA_HOST=http://localhost:11434
```

---

## Quota Solution

### For Gemini Free Tier:
- **Limit:** 60 requests/minute, 1500 requests/day
- **Solution:** Add delays between requests (1-2 seconds)

### For OpenAI:
- Use **GPT-4o-mini** (cheapest: $0.15/1M input tokens)
- Estimated cost for thesis: ~$0.50 total

### For Ollama (Llama, Qwen):
- **No limits!** Run as many requests as you want
- Completely free and private

---

## System Requirements for Ollama

**Minimum:**
- 8GB RAM
- 10GB free disk space

**Recommended:**
- 16GB RAM
- GPU (NVIDIA) for faster inference
- 20GB free disk space

**Without GPU:** Models will run on CPU (slower but works fine)

---

## Next Steps

After installing Ollama and pulling models, I will:
1. Implement OpenAI client
2. Implement Ollama client (works for Llama, Qwen, Mistral)
3. Update persona registry with provider assignments
4. Add rate limiting to avoid quota issues
5. Test the multi-provider MAD system
