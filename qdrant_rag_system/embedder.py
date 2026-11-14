
import os
import base64
from openai import OpenAI
from PIL import Image
from io import BytesIO


class EuriEmbedder:

    def __init__(self, model_name="euri-embedding-multimodal"):

        api_key = os.environ.get("EURI_API_KEY")

        if not api_key:
            raise ValueError("EURI_API_KEY environment variable not set.")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.euron.one/api/v1/euri"
        )

        self.model_name = model_name

    def embed_image(self, image: Image.Image) -> list[float]:

        print(f"Embedding image using model: {self.model_name}...")

        try:
            buffered = BytesIO()
            image.save(buffered, format="PNG")

            image_base64 = base64.b64encode(
                buffered.getvalue()
            ).decode("utf-8")

            result = self.client.embeddings.create(
                model=self.model_name,
                input=[
                    {
                        "image": f"data:image/png;base64,{image_base64}"
                    }
                ]
            )

            return result.data[0].embedding

        except Exception as e:
            print(f"Error embedding image: {e}")
            raise

    def embed_text(self, text: str) -> list[float]:

        print(f"Embedding text using model: {self.model_name}...")

        try:
            result = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )

            return result.data[0].embedding

        except Exception as e:
            print(f"Error embedding text: {e}")
            raise
