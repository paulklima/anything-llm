const { toChunks } = require("../../helpers");

class MistralEmbedder {
  constructor() {
    if (!process.env.MISTRAL_API_KEY)
      throw new Error("No Mistral API key was set.");

    const { MistralAIEmbeddings } = require("@langchain/mistralai");
    const embeddings = new MistralAIEmbeddings({
      apiKey: process.env.MISTRAL_API_KEY,
    });

    this.mistral = embeddings;

    // Limit of how many strings we can process in a single pass to stay with resource or network limits
    this.maxConcurrentChunks = 2;

    // Mistral max token limit per request is 16_384 tokens. Set to 16k to be safe for keep space for Metadata.
    this.embeddingMaxChunkLength = 16_000;
  }

  async embedTextInput(textInput) {
    const result = await this.embedChunks(
      Array.isArray(textInput) ? textInput : [textInput]
    );
    return result?.[0] || [];
  }

  async embedChunks(textChunks = []) {
    const embeddingRequests = [];

    const chunkedText = toChunks(textChunks, this.maxConcurrentChunks);

    for (const chunk of chunkedText) {
      embeddingRequests.push(
        new Promise((resolve) => {
          this.mistral
            .embedDocuments(chunk)
            .then((result) => {
              resolve({ data: result, error: null });
            })
            .catch((e) => {
              console.error("Mistral returns error: ", e);
              e.type =
                e?.response?.data?.error?.code ||
                e?.response?.status ||
                "failed_to_embed";
              e.message = e?.response?.data?.error?.message || e.message;
              resolve({ data: [], error: e });
            });
        })
      );
    }

    const { data = [], error = null } = await Promise.all(
      embeddingRequests
    ).then((results) => {
      // If any errors were returned from OpenAI abort the entire sequence because the embeddings
      // will be incomplete.
      const errors = results
        .filter((res) => !!res.error)
        .map((res) => res.error)
        .flat();
      if (errors.length > 0) {
        console.log("Mistral errors: ", errors.length);
        let uniqueErrors = new Set();
        errors.map((error) =>
          uniqueErrors.add(`[${error.type}]: ${error.message}`)
        );

        return {
          data: [],
          error: Array.from(uniqueErrors).join(", "),
        };
      }
      return {
        data: results.map((res) => res?.data).flat(),
        error: null,
      };
    });

    if (!!error) throw new Error(`Mistral Failed to embed: ${error}`);
    return data?.length > 0 ? data : null;
  }
}

module.exports = {
  MistralEmbedder,
};
