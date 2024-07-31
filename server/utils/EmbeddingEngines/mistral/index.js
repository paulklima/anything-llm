const { toChunks } = require("../../helpers");

class MistralEmbedder {
  constructor() {
    if (!process.env.MISTRAL_API_KEY)
      throw new Error("No Mistral API key was set.");
    // const { OpenAI: OpenAIApi } = require("openai");
    // this.openai = new OpenAIApi({
    //   baseURL: "https://api.mistral.ai/v1",
    //   apiKey: process.env.MISTRAL_API_KEY,
    // });
    // this.model = process.env.EMBEDDING_MODEL_PREF || "mistral-embed";

    const { MistralAIEmbeddings } = require("@langchain/mistralai");
    const embeddings = new MistralAIEmbeddings({
      apiKey: process.env.MISTRAL_API_KEY,
    });

    this.mistral = embeddings;

    // Limit of how many strings we can process in a single pass to stay with resource or network limits
    // this.maxConcurrentChunks = 500;
    this.maxConcurrentChunks = 100;

    // https://platform.openai.com/docs/guides/embeddings/embedding-models
    this.embeddingMaxChunkLength = 8_191;
    // this.embeddingMaxChunkLength = 400;
  }


  // async embedTextInput(textInput) {
  //   console.log("embedTextInput", textInput.length);
  //   const result = await this.mistral.embedDocuments(
  //     Array.isArray(textInput) ? textInput : [textInput],
  //   );

  //   // If given an array return the native Array[Array] format since that should be the outcome.
  //   // But if given a single string, we need to flatten it so that we have a 1D array.
  //   return (Array.isArray(textInput) ? result : result.flat()) || [];
  // }

  async embedTextInput(textInput) {
    console.log("embedTextInput", textInput.length);
    const result = await this.embedChunks(
      Array.isArray(textInput) ? textInput : [textInput]
    );
    return result?.[0] || [];
  }

  async embedChunks(textChunks = []) {
    console.log("embedChunks: ", textChunks.length);

    const embeddingRequests = [];

    const chunkedText = toChunks(textChunks, this.maxConcurrentChunks);
    console.log("chunkedText: ", chunkedText.length);
    console.log("example: ", chunkedText[0]);
    for (const chunk of chunkedText) {

      // let charLength = 0;
      // for(const c of chunk) {
      //   charLength += !!c ? 0 : c.length;
      // }
      // console.log("chunk: ", chunk.array.length, charLength);

      embeddingRequests.push(
        new Promise((resolve) => {
          this.mistral
            .embedDocuments(chunk)
            .then((result) => {
              console.log("Mistral API result: ", result);
              resolve({ data: result?.data, error: null });
            })
            .catch((e) => {
              console.error("Mistral error: ", e);
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
      console.log("Mistral results: ", results.length);
      console.log("Mistral Results: ", results[0]);
      return results;
    });

    if (!!error) throw new Error(`Mistral Failed to embed: ${error}`);
    return data.length > 0 ? data.map((embd) => embd.embedding) : null;
  }
}

module.exports = {
  MistralEmbedder,
};
