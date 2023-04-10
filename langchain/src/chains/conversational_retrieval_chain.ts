import { PromptTemplate } from "../prompts/index.js";
import { BaseLLM } from "../llms/index.js";
import { SerializedChatVectorDBQAChain } from "./serde.js";
import { ChainValues, BaseRetriever } from "../schema/index.js";
import { BaseChain } from "./base.js";
import { LLMChain } from "./llm_chain.js";
import {
  loadQAMapReduceChain,
  loadQAStuffChain,
} from "./question_answering/load.js";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type LoadValues = Record<string, any>;

const question_generator_template = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

export interface ConversationalRetrievalQAChainInput {
  retriever: BaseRetriever;
  combineDocumentsChain: BaseChain;
  questionGeneratorChain: LLMChain;
  outputKey: string;
  inputKey: string;
}

export class ConversationalRetrievalQAChain
  extends BaseChain
  implements ConversationalRetrievalQAChainInput
{
  inputKey = "question";

  chatHistoryKey = "chat_history";

  get inputKeys() {
    return [this.inputKey, this.chatHistoryKey];
  }

  outputKey = "result";

  retriever: BaseRetriever;

  combineDocumentsChain: BaseChain;

  questionGeneratorChain: LLMChain;

  returnSourceDocuments = false;

  constructor(fields: {
    retriever: BaseRetriever;
    combineDocumentsChain: BaseChain;
    questionGeneratorChain: LLMChain;
    inputKey?: string;
    outputKey?: string;
    returnSourceDocuments?: boolean;
  }) {
    super();
    this.retriever = fields.retriever;
    this.combineDocumentsChain = fields.combineDocumentsChain;
    this.questionGeneratorChain = fields.questionGeneratorChain;
    this.inputKey = fields.inputKey ?? this.inputKey;
    this.outputKey = fields.outputKey ?? this.outputKey;
    this.returnSourceDocuments =
      fields.returnSourceDocuments ?? this.returnSourceDocuments;
  }

  async _call(values: ChainValues): Promise<ChainValues> {
    if (!(this.inputKey in values)) {
      throw new Error(`Question key ${this.inputKey} not found.`);
    }
    if (!(this.chatHistoryKey in values)) {
      throw new Error(`chat history key ${this.inputKey} not found.`);
    }
    const question: string = values[this.inputKey];
    const chatHistory: string = values[this.chatHistoryKey];
    let newQuestion = question;
    if (chatHistory.length > 0) {
      const result = await this.questionGeneratorChain.call({
        question,
        chat_history: chatHistory,
      });
      const keys = Object.keys(result);
      if (keys.length === 1) {
        newQuestion = result[keys[0]];
      } else {
        throw new Error(
          "Return from llm chain has multiple values, only single values supported."
        );
      }
    }
    const docs = await this.retriever.getRelevantDocuments(newQuestion);
    const inputs = {
      question: newQuestion,
      input_documents: docs,
      chat_history: chatHistory,
    };
    const result = await this.combineDocumentsChain.call(inputs);
    if (this.returnSourceDocuments) {
      return {
        ...result,
        sourceDocuments: docs,
      };
    }
    return result;
  }

  _chainType(): string {
    return "conversational_retrieval_chain";
  }

  static async deserialize(
    _data: SerializedChatVectorDBQAChain,
    _values: LoadValues
  ): Promise<ConversationalRetrievalQAChain> {
    throw new Error("Not implemented.");
  }

  serialize(): SerializedChatVectorDBQAChain {
    throw new Error("Not implemented.");
  }

  static fromLLM(
    llm: BaseLLM,
    retriever: BaseRetriever,
    options: {
      inputKey?: string;
      outputKey?: string;
      returnSourceDocuments?: boolean;
      questionGeneratorTemplate?: string;
      qaTemplate?: string;
      chainType?: string;
    } = {}
  ): ConversationalRetrievalQAChain {
    const {
      questionGeneratorTemplate,
      qaTemplate,
      chainType = "stuff_documents_chain",
      ...rest
    } = options;
    const question_generator_prompt = PromptTemplate.fromTemplate(
      questionGeneratorTemplate || question_generator_template
    );

    let qaChain: BaseChain;

    const qaPrompt = qaTemplate
      ? PromptTemplate.fromTemplate(qaTemplate)
      : undefined;

    if (chainType === "stuff_documents_chain") {
      qaChain = loadQAStuffChain(llm, { prompt: qaPrompt });
    } else if (chainType === "map_reduce_documents_chain") {
      qaChain = loadQAMapReduceChain(llm, { combineMapPrompt: qaPrompt });
    } else {
      throw new Error(
        `Unknown chain type ${chainType}. Chain type should be one of the following: stuff_documents_chain, map_reduce_documents_chain.`
      );
    }
    const questionGeneratorChain = new LLMChain({
      prompt: question_generator_prompt,
      llm,
    });
    const instance = new this({
      retriever,
      combineDocumentsChain: qaChain,
      questionGeneratorChain,
      ...rest,
    });
    return instance;
  }
}
