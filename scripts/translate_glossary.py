#!/usr/bin/env python3
"""Translate glossary/terms.md using the API, preserving structure."""
import json, re, ssl, time, urllib.request

API_BASE = "https://token-plan-cn.xiaomimimo.com/v1"
API_KEY = "tp-cbsgysth0q8iuh5e9e81ha9uzxdc4zdrrxfbh7z9gp60zz3e"
MODEL = "mimo-v2.5-pro"
CTX = ssl._create_unverified_context()

def translate(text):
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a technical translator. Translate the following glossary entry to Chinese. Preserve ALL markdown formatting (###, **, -, etc). Keep technical terms like LLM, GPU, ReLU, Adam, etc. in their original English form. Translate the descriptions. Do NOT add any commentary."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.2, "max_tokens": 2048
    }).encode()
    req = urllib.request.Request(f"{API_BASE}/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]

def main():
    inpath = "/Users/xiaojia/code/ai-engineering-from-scratch/glossary/terms.md"
    outpath = "/Users/xiaojia/code/ai-engineering-from-scratch/glossary/terms-zh.md"
    
    with open(inpath) as f:
        content = f.read()
    
    # Split into blocks: ## X header + ### Term entries
    # Parse into sections
    header_match = re.match(r'(# AI Engineering Glossary\n)', content)
    header = header_match.group(1) if header_match else "# AI 工程术语表\n"
    
    # Split by "## " section headers
    sections = re.split(r'\n(?=## )', content)
    
    translated_sections = []
    # First section is the intro (no ##)
    if sections[0].startswith("# "):
        translated_sections.append("# AI 工程术语表\n")
        sections = sections[1:]
    else:
        translated_sections.append(sections[0])
        sections = sections[1:]
    
    total_terms = 0
    for section in sections:
        # Each section has ## X header + ### term entries
        # Translate section header
        lines = section.strip().split('\n', 1)
        if len(lines) < 2:
            translated_sections.append(section)
            continue
        
        section_header = lines[0]  # e.g., "## A"
        section_body = lines[1]
        
        # Translate section header letter (keep "## " prefix)
        translated_sections.append(section_header + '\n')
        
        # Split section body into term blocks by "### "
        term_blocks = re.split(r'\n(?=### )', section_body)
        
        for block in term_blocks:
            if not block.strip():
                continue
            # Keep ### Term name as-is, translate the rest
            lines2 = block.strip().split('\n', 1)
            term_header = lines2[0]  # "### Attention"
            if len(lines2) > 1:
                term_body = lines2[1]
                # Translate the body
                to_translate = term_body.strip()
                try:
                    translated_body = translate(to_translate)
                    translated_sections.append(term_header + '\n' + translated_body + '\n\n')
                    total_terms += 1
                    print(f"  [{total_terms}] {term_header}")
                except Exception as e:
                    print(f"  [{total_terms}] {term_header} FAILED: {e}")
                    translated_sections.append(block + '\n\n')
                time.sleep(0.5)
            else:
                translated_sections.append(block + '\n\n')
    
    result = ''.join(translated_sections)
    with open(outpath, 'w') as f:
        f.write(result)
    print(f"\nDone! {total_terms} terms translated -> {outpath}")

if __name__ == "__main__":
    main()
