
from services.embedding_service import get_embedding_model

def main():

    embeddings = get_embedding_model()

    text = "What is Generative AI?"

    vector = embeddings.embed_query(text)

    print("Original text:", text)

    print("Vector dimensions:", len(vector))

    print("First 10 vector values:", vector[:10])


if __name__ == "__main__":
    main()