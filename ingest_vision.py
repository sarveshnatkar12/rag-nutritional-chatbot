# ingest.py


# ╔══════════════════════════════════════════════════════════╗
#   In this code, we are doing the following:
#   1. Read PDF file
#   2. Convert PDF into sentences AND extract images
#   3. Group 20 sentences into 1 chunk
#   4. Collect all chunks
#   5. Convert each chunk into a vector embedding
#   6. Extract images and convert to embeddings using vision model
#   7. Upload the chunks and embeddings to Supabase
# ╚══════════════════════════════════════════════════════════╝


#pip install -q pymupdf tiktoken supabase openai tqdm python-dotenv pillow


import os, uuid, re, base64, io
import fitz  # PyMuPDF
import tiktoken
from supabase import create_client, Client
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv, find_dotenv
from PIL import Image


# ---- Load environment
load_dotenv(find_dotenv(usecwd=True))


SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


# ---- Config
PDF_PATH = "human-nutrition-text.pdf"
DOC_ID = "nutrition-v1"               # keep this STABLE to avoid duplicates
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims -> matches your table
VISION_MODEL = "gpt-4o"              # for image analysis and description
BATCH_EMBED = 100
BATCH_INSERT = 200


# Sentence chunking params
SENTS_PER_CHUNK = 20
SENT_OVERLAP = 2
MAX_TOKENS = 1300     # safety cap (trim if 10 sentences are too long)
MIN_TOKENS = 50      # skip very tiny fragments


# Image processing params
MIN_IMAGE_SIZE = 100  # minimum width/height in pixels
MAX_IMAGE_SIZE = 2048 # resize larger images
IMAGE_QUALITY = 85    # JPEG quality for compression


enc = tiktoken.get_encoding("cl100k_base")  # matches OpenAI embeddings tokenizer


def clean_text(t: str) -> str:
   # normalize whitespace and fix hyphenation across line breaks
   t = t.replace("\r", " ")
   t = re.sub(r"-\s*\n\s*", "", t)     # join "nutri-\n tion" => "nutrition"
   t = re.sub(r"\s+\n", "\n", t)
   t = re.sub(r"[ \t]+", " ", t)
   t = t.replace("\n", " ").strip()
   return t


def split_sentences(text: str):
   # simple sentence splitter (good for prose)
   sents = re.split(r'(?<=[.!?])\s+', text.strip())
   return [s.strip() for s in sents if s.strip()]


def chunk_page_by_sentences(text: str,
                           sents_per_chunk: int = SENTS_PER_CHUNK,
                           overlap: int = SENT_OVERLAP,
                           max_tokens: int = MAX_TOKENS,
                           min_tokens: int = MIN_TOKENS):
   sents = split_sentences(text)
   i = 0
   step = max(1, sents_per_chunk - overlap)
   while i < len(sents):
       piece = sents[i:i + sents_per_chunk]
       if not piece:
           break
       chunk = " ".join(piece)


       # enforce token ceiling
       ids = enc.encode(chunk)
       while max_tokens and len(ids) > max_tokens and len(piece) > 1:
           piece = piece[:-1]
           chunk = " ".join(piece)
           ids = enc.encode(chunk)


       if len(ids) >= min_tokens:
           yield chunk
       i += step


def process_image(img_data, max_size=MAX_IMAGE_SIZE):
   """Process image: resize if needed and convert to base64."""
   try:
       # Load image
       img = Image.open(io.BytesIO(img_data))
      
       # Convert to RGB if necessary
       if img.mode in ('RGBA', 'LA', 'P'):
           img = img.convert('RGB')
      
       # Check minimum size
       if img.width < MIN_IMAGE_SIZE or img.height < MIN_IMAGE_SIZE:
           return None
      
       # Resize if too large
       if img.width > max_size or img.height > max_size:
           ratio = min(max_size/img.width, max_size/img.height)
           new_size = (int(img.width * ratio), int(img.height * ratio))
           img = img.resize(new_size, Image.Resampling.LANCZOS)
      
       # Convert to base64
       buffer = io.BytesIO()
       img.save(buffer, format='JPEG', quality=IMAGE_QUALITY)
       img_b64 = base64.b64encode(buffer.getvalue()).decode()
      
       return img_b64, img.width, img.height
  
   except Exception as e:
       print(f"Error processing image: {e}")
       return None


def extract_images_from_page(page):
   """Extract images from a PDF page."""
   images = []
   img_list = page.get_images()
  
   for img_idx, img in enumerate(img_list):
       try:
           # Get image data
           xref = img[0]
           pix = fitz.Pixmap(page.parent, xref)
          
           # Skip if image is too small or has issues
           if pix.width < MIN_IMAGE_SIZE or pix.height < MIN_IMAGE_SIZE:
               pix = None
               continue
          
           # Convert to PIL format
           if pix.n - pix.alpha < 4:  # GRAY or RGB
               img_data = pix.tobytes("png")
           else:  # CMYK
               pix1 = fitz.Pixmap(fitz.csRGB, pix)
               img_data = pix1.tobytes("png")
               pix1 = None
          
           pix = None
          
           # Process image
           processed = process_image(img_data)
           if processed:
               img_b64, width, height = processed
               images.append({
                   'data': img_b64,
                   'width': width,
                   'height': height,
                   'index': img_idx
               })
      
       except Exception as e:
           print(f"Error extracting image {img_idx}: {e}")
           continue
  
   return images


def get_image_description(client, img_b64):
   """Get description of image using vision model."""
   try:
       response = client.chat.completions.create(
           model=VISION_MODEL,
           messages=[
               {
                   "role": "user",
                   "content": [
                       {
                           "type": "text",
                           "text": "Describe this image in detail, focusing on any text, diagrams, charts, tables, or educational content. Be specific about what information this image conveys."
                       },
                       {
                           "type": "image_url",
                           "image_url": {
                               "url": f"data:image/jpeg;base64,{img_b64}"
                           }
                       }
                   ]
               }
           ],
           max_tokens=500
       )
       return response.choices[0].message.content
   except Exception as e:
       print(f"Error getting image description: {e}")
       return "Image content could not be analyzed."


def pdf_pages_and_images(path: str):
   """Yield (page_number_1based, cleaned_text, images)."""
   doc = fitz.open(path)
   try:
       for i in range(len(doc)):
           page = doc[i]
           txt = page.get_text("text") or ""
           images = extract_images_from_page(page)
           yield (i + 1, clean_text(txt), images)
   finally:
       doc.close()


def main():
   sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
   client = OpenAI(api_key=OPENAI_API_KEY)


   # Optional: keep the table clean for this document
   sb.table("chunks").delete().eq("doc_id", DOC_ID).execute()


   print("Reading PDF by pages and extracting images...")
   pages_data = list(pdf_pages_and_images(PDF_PATH))


   # Build text chunks and image data
   text_inputs, text_metas = [], []
   image_inputs, image_metas, image_descriptions = [], [], []
  
   print("Processing text chunks and images...")
   total_images = 0
  
   for page_num, text, images in pages_data:
       # Process text chunks
       if text:
           for chunk in chunk_page_by_sentences(text):
               text_inputs.append(chunk)
               text_metas.append({
                   "page": page_num,
                   "source": PDF_PATH,
                   "type": "text"
               })
      
       # Process images
       for img_data in images:
           print(f"Processing image {img_data['index']} from page {page_num}...")
           description = get_image_description(client, img_data['data'])
          
           # Use description as the "content" for embedding
           image_inputs.append(description)
           image_metas.append({
               "page": page_num,
               "source": PDF_PATH,
               "type": "image",
               "image_index": img_data['index'],
               "width": img_data['width'],
               "height": img_data['height'],
               "image_data": img_data['data']  # Store base64 image data
           })
           image_descriptions.append(description)
           total_images += 1


   print(f"Built {len(text_inputs)} text chunks and {len(image_inputs)} image chunks from {PDF_PATH}")
   print(f"Total images processed: {total_images}")


   # Combine all inputs for embedding
   all_inputs = text_inputs + image_inputs
   all_metas = text_metas + image_metas


   if not all_inputs:
       print("No content to process!")
       return


   # Generate embeddings for all content (text + image descriptions)
   vectors = []
   print("Generating embeddings...")
   for i in tqdm(range(0, len(all_inputs), BATCH_EMBED), desc="Embedding"):
       batch = all_inputs[i:i + BATCH_EMBED]
       resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
       vectors.extend([d.embedding for d in resp.data])


   # Prepare rows
   rows = []
   for idx, (content, emb, meta) in enumerate(zip(all_inputs, vectors, all_metas)):
       row = {
           "doc_id": DOC_ID,
           "chunk_index": idx,
           "content": content,
           "metadata": meta,
           "embedding": emb
       }
       rows.append(row)


   print("Uploading to Supabase...")
   for j in tqdm(range(0, len(rows), BATCH_INSERT), desc="Uploading"):
       sb.table("chunks").insert(rows[j:j+BATCH_INSERT]).execute()


   print(f"✅ Done! Inserted {len(rows)} chunks for doc_id={DOC_ID}")
   print(f"   - Text chunks: {len(text_inputs)}")
   print(f"   - Image chunks: {len(image_inputs)}")


if __name__ == "__main__":
   main()
