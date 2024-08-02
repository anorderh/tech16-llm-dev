import os
import requests;
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from youtubesearchpython import Video, ResultMode
from google.colab import userdata
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.readers.file import ImageReader
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import PromptTemplate

# Setup creds.
open_ai_key = userdata.get('open_ai_key')
os.environ["OPENAI_API_KEY"] = open_ai_key

# Setup image storage.
parser = ImageReader()
file_extractor = {
    ".jpg": parser,
    ".jpeg": parser,
}
thumbnail_dir = "./thumbnails"
curr_thumbnail_path = './thumbnails/curr.jpg';
if not os.path.exists(thumbnail_dir):
    os.makedirs(thumbnail_dir)

# Setup LlamaIndex params.
client = qdrant_client.QdrantClient(":memory:")
image_store = QdrantVectorStore(
    client=client, collection_name="images"
)
storage_context = StorageContext.from_defaults(image_store=image_store)
qa_tmpl_str = (
  "Given the images provided, "
  "answer the query.\n"
  "Query: {query_str}\n"
  "Answer: "
)
qa_tmpl = PromptTemplate(qa_tmpl_str)
openai_mm_llm = OpenAIMultiModal(
    model="gpt-4o", api_key=open_ai_key, max_new_tokens=1500
)

def downloadThumbnail(videoId):
  thumbnail_url = f"https://i.ytimg.com/vi/{videoId}/hq720.jpg"
  img_data = requests.get(thumbnail_url).content
  with open(curr_thumbnail_path, 'wb') as handler:
      handler.write(img_data);

  img = mpimg.imread(curr_thumbnail_path)
  plt.imshow(img)
  plt.axis('off')
  plt.show()

def procureQueryEngine():
  documents = SimpleDirectoryReader(
      thumbnail_dir,
      file_extractor=file_extractor
  ).load_data()
  index = MultiModalVectorStoreIndex.from_documents(
      documents,
      storage_context=storage_context,
  )
  query_engine = index.as_query_engine(
      llm=openai_mm_llm, image_qa_template=qa_tmpl
  )

  return query_engine;


### Provide a video ID and ask questions.
'''
Channel:
**TechLinked** - News about tech & gaming culture, delivered thrice weekly

Video:
*Do we even need ARM?* - https://www.youtube.com/watch?v=-8wDDeCcFhc
'''
videoId = "-8wDDeCcFhc"
downloadThumbnail(videoId);

engine = procureQueryEngine()

response = engine.image_query(
    curr_thumbnail_path,
    "What can you tell me about this image?"
)
print(response)

response = engine.image_query(
    curr_thumbnail_path,
    "What can you tell me about the demographic this image is appealing to?"
)
print(response)

response = engine.image_query(
    curr_thumbnail_path,
    "This image is a Youtube thumbnail. How would you suggest I change it to improve its click through rate?"
)
print(response)