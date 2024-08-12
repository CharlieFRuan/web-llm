/**
 * The input to OpenAI API, directly adopted from openai-node with small tweaks:
 * https://github.com/openai/openai-node/blob/master/src/resources/embeddings.ts
 *
 * Copyright 2024 OpenAI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *      http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  EmbeddingInputEmptyError,
  EmbeddingUnsupportedEncodingFormatError,
  UnsupportedFieldsError,
} from "../error";
import { MLCEngineInterface } from "../types";

export class Embeddings {
  private engine: MLCEngineInterface;

  constructor(engine: MLCEngineInterface) {
    this.engine = engine;
  }

  /**
   * Creates an embedding vector representing the input text.
   */
  create(request: EmbeddingCreateParams): Promise<CreateEmbeddingResponse> {
    return this.engine.embedding(request);
  }
}

export interface CreateEmbeddingResponse {
  /**
   * The list of embeddings generated by the model.
   */
  data: Array<Embedding>;

  /**
   * The name of the model used to generate the embedding.
   */
  model: string;

  /**
   * The object type, which is always "list".
   */
  object: "list";

  /**
   * The usage information for the request.
   */
  usage: CreateEmbeddingResponse.Usage;
}

/* eslint-disable-next-line @typescript-eslint/no-namespace */
export namespace CreateEmbeddingResponse {
  /**
   * The usage information for the request.
   */
  export interface Usage {
    /**
     * The number of tokens used by the prompt.
     */
    prompt_tokens: number;

    /**
     * The total number of tokens used by the request.
     */
    total_tokens: number;

    /**
     * Fields specific to WebLLM, not present in OpenAI.
     */
    extra: {
      /**
       * Number of tokens per second for prefilling.
       */
      prefill_tokens_per_s: number;
    };
  }
}

/**
 * Represents an embedding vector returned by embedding endpoint.
 */
export interface Embedding {
  /**
   * The embedding vector, which is a list of floats. The length of vector depends on
   * the model.
   */
  embedding: Array<number>;

  /**
   * The index of the embedding in the list of embeddings.
   */
  index: number;

  /**
   * The object type, which is always "embedding".
   */
  object: "embedding";
}

export interface EmbeddingCreateParams {
  /**
   * Input text to embed, encoded as a string or array of tokens. To embed multiple
   * inputs in a single request, pass an array of strings or array of token arrays.
   * The input must not exceed the max input tokens for the model, and cannot be an empty string.
   * If the batch size is too large, multiple forward of the will take place.
   */
  input: string | Array<string> | Array<number> | Array<Array<number>>;

  /**
   * The format to return the embeddings in.
   *
   * @note Currently only support `float`.
   */
  encoding_format?: "float" | "base64";

  /**
   * ID of the model to use.
   *
   * @note Not supported. Instead, call `CreateMLCEngine(model)` or `engine.reload(model)`.
   */
  model?: string;

  // TODO: can support matryoshka embedding models in future, hence allow `dimensions` for those.
  /**
   * The number of dimensions the resulting output embeddings should have.
   *
   * @note Not supported.
   */
  dimensions?: number;

  /**
   * A unique identifier representing your end-user, which can help OpenAI to monitor
   * and detect abuse.
   *
   * @note Not supported.
   */
  user?: string;
}

export const EmbeddingCreateParamsUnsupportedFields: Array<string> = [
  "model",
  "dimensions",
  "user",
];

export function postInitAndCheckFields(
  request: EmbeddingCreateParams,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  currentModelId: string,
): void {
  // 1. Check unsupported fields in request
  const unsupported: Array<string> = [];
  EmbeddingCreateParamsUnsupportedFields.forEach((field) => {
    if (field in request) {
      unsupported.push(field);
    }
  });
  if (unsupported.length > 0) {
    throw new UnsupportedFieldsError(unsupported, "EmbeddingCreateParams");
  }

  // 2. Unsupported format
  if (request.encoding_format == "base64") {
    throw new EmbeddingUnsupportedEncodingFormatError();
  }

  // 3. Invalid input
  const input = request.input;
  if (typeof input === "string") {
    if (input === "") throw new EmbeddingInputEmptyError();
  } else {
    // input instanceof Array
    if (input.length === 0) {
      // Array<number>
      throw new EmbeddingInputEmptyError();
    }
    for (let i = 0; i < input.length; i++) {
      const curInput = input[i];
      if (typeof curInput !== "number") {
        // Array<string>, Array<Array<number>>
        if (curInput.length === 0) throw new EmbeddingInputEmptyError();
      }
    }
  }
}
