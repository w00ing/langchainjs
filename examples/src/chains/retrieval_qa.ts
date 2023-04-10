import { OpenAI } from "langchain/llms";
import { RetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

export const run = async () => {
  // Initialize the LLM to use to answer the question.
  const model = new OpenAI({});
  const text = fs.readFileSync("state_of_the_union.txt", "utf8");
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);

  // Create a vector store from the documents.
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

  // Create a chain that uses the OpenAI LLM and HNSWLib vector store.
  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());
  const res = await chain.call({
    query: "What did the president say about Justice Breyer?",
  });
  console.log({ res });
  /*
  {
    res: {
      text: 'The president said that Justice Breyer was an Army veteran, Constitutional scholar,
      and retiring Justice of the United States Supreme Court and thanked him for his service.'
    }
  }
  */

  // You can also use Map Reduce chain method
  const mapReduceChain = RetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever(),
    { chainType: "map_reduce_documents_chain" }
  );

  const mapReduceChainRes = await mapReduceChain.call({
    query: "What did the president say about Justice Breyer?",
  });
  console.log(mapReduceChainRes);
};
