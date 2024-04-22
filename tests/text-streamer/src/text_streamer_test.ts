import { TextStreamer } from "../../../src/streamer";
import * as tvmjs from "tvmjs";
import { Tokenizer } from "@mlc-ai/web-tokenizers";

// tests' inputs and outputs
const llama2_emoji_tokens_expected_result: [number[], string[]][] = [
  // HF: "�����", SentencePiece: "�👀"
  [[177, 243, 162, 148, 131], ["�����", "�👀"]],
  // Both: "👀👀"
  [[243, 162, 148, 131, 243, 162, 148, 131], ["👀👀",]],
  // Both: "👀👀👀"
  [[243, 162, 148, 131, 243, 162, 148, 131, 243, 162, 148, 131], ["👀👀👀",]],
  // HF: "👀�������", SentencePiece: "👀���👀"
  [[243, 162, 148, 131, 162, 148, 131, 243, 162, 148, 131], ["👀�������", "👀���👀"]],
  // Both: "👀��� have👀"
  [[243, 162, 148, 131, 162, 148, 131, 505, 243, 162, 148, 131], ["👀��� have👀",]],
];

const llama3_emoji_tokens_expected_result: [number[], string[]][] = [
  [[222, 9468, 239, 222], ["�����", "�👀"]],
  [[9468, 239, 222, 9468, 239, 222], ["👀👀",]],
  // [[243, 162, 148, 131, 243, 162, 148, 131, 243, 162, 148, 131], ["👀👀👀",]],
  // [[243, 162, 148, 131, 162, 148, 131, 243, 162, 148, 131], ["👀�������", "👀���👀"]],
  // [[243, 162, 148, 131, 162, 148, 131, 505, 243, 162, 148, 131], ["👀��� have👀",]],
];

const llama2_para_input_tokens = [18585, 29892, 1244, 29915, 29879, 263, 3273, 14880, 1048, 953, 29877, 2397,
  29892, 988, 1269, 1734, 338, 5643, 491, 385, 953, 29877, 2397, 29901, 13, 13,
  29950, 1032, 727, 29991, 29871, 243, 162, 148, 142, 306, 29915, 29885, 1244, 304,
  1371, 1234, 738, 5155, 366, 505, 1048, 953, 29877, 2397, 29871, 243, 162, 167, 151,
  29889, 7440, 366, 1073, 393, 953, 29877, 2397, 508, 367, 1304, 304, 27769, 23023,
  1080, 322, 21737, 297, 263, 2090, 322, 1708, 1319, 982, 29973, 29871, 243, 162, 155,
  135, 2688, 508, 884, 367, 1304, 304, 788, 263, 6023, 310, 2022, 2877, 304, 596, 7191,
  322, 11803, 29889, 29871, 243, 162, 149, 152, 1126, 29892, 1258, 366, 1073, 393, 727,
  526, 1584, 953, 29877, 2397, 8090, 322, 14188, 366, 508, 1708, 29973, 29871, 243, 162,
  145, 177, 243, 162, 148, 131, 1105, 29892, 748, 14432, 322, 679, 907, 1230, 411, 953,
  29877, 2397, 29991, 29871, 243, 162, 149, 168, 243, 162, 145, 171]

const llama3_para_input_tokens = [40914, 11, 1618, 596, 264, 2875, 14646, 922, 43465, 11, 1405, 1855,
  3492, 374, 8272, 555, 459, 43465, 1473, 19182, 1070, 0, 62904, 233, 358, 2846, 1618, 311, 1520,
  4320, 904, 4860, 499, 617, 922, 43465, 11410, 97, 242, 13, 14910, 499, 1440, 430, 43465, 649,
  387, 1511, 311, 20599, 21958, 323, 16024, 304, 264, 2523, 323, 57169, 1648, 30, 27623, 226, 2435,
  649, 1101, 387, 1511, 311, 923, 264, 5916, 315, 17743, 311, 701, 6743, 323, 8158, 13, 64139, 243,
  1628, 11, 1550, 499, 1440, 430, 1070, 527, 1524, 43465, 3953, 323, 7640, 499, 649, 1514, 30,
  11410, 236, 106, 9468, 239, 222, 2100, 11, 733, 8469, 323, 636, 11782, 449, 43465, 0, 64139, 98,
  9468, 236, 101];

const DECODED_PARAGRAPH = (
  "Sure, here's a short paragraph about emoji, " +
  "where each word is followed by an emoji:\n\n" +
  "Hey there! 👋 I'm here to help answer any questions you have about emoji 🤔. " +
  "Did you know that emoji can be used to convey emotions and feelings in a " +
  "fun and playful way? 😄 " +
  "They can also be used to add a touch of personality to your messages and posts. 💕 " +
  "And, did you know that there are even emoji games and activities you can play? 🎮👀 " +
  "So, go ahead and get creative with emoji! 💥🎨"
)

async function loadTokenizer(useJSON: boolean, useLlama3: boolean) {
  const modelCache = new tvmjs.ArtifactCache("webllm/model");
  let baseUrl;
  if (useLlama3) {
    baseUrl = "https://huggingface.co/mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC/resolve/main/";
  } else {
    baseUrl = "https://huggingface.co/mlc-ai/Llama-2-7b-chat-hf-q4f16_1-MLC/resolve/main/";
  }
  let tokenizer: Tokenizer;
  if (useJSON) {
    const url = new URL("tokenizer.json", baseUrl).href;
    const model = await (await modelCache.fetchWithCache(url)).arrayBuffer();
    tokenizer = await Tokenizer.fromJSON(model);
  } else {
    const url = new URL("tokenizer.model", baseUrl).href;
    const model = await modelCache.fetchWithCache(url, "arraybuffer");
    tokenizer = await Tokenizer.fromSentencePiece(model);
  }
  return tokenizer;
}


// Tests are below
async function test_text_streamer_emojis(useJSON: boolean, useLlama3: boolean) {
  const tokenizer = await loadTokenizer(useJSON, useLlama3);
  const emoji_tokens_expected_result = useLlama3 ?
    llama3_emoji_tokens_expected_result : llama2_emoji_tokens_expected_result;

  for (const test_i of emoji_tokens_expected_result) {
    const tokens = test_i[0];
    const expected_results = test_i[1];
    const textStreamer = new TextStreamer(tokenizer);
    let total_text = "";
    for (const token of tokens) {
      total_text += textStreamer.put([token]);
    }
    total_text += textStreamer.finish();
    if (!(expected_results.includes(total_text))) {
      throw Error("Got " + total_text + "while expect one of: " + expected_results);
    }
  }
  console.log("Passed test_text_streamer_emojis, useJSON=" + useJSON + ", useLlama3=" + useLlama3);
}

async function test_text_streamer(useJSON: boolean, useLlama3: boolean) {
  // TO BE REMOVED
  const llama3Tokenizer = await loadTokenizer(true, true);
  const encoded_para = llama3Tokenizer.encode(DECODED_PARAGRAPH);
  console.log(llama3Tokenizer.encode("👍🏽"));  // 9468, 239, 235, 9468, 237, 121
  console.log(llama3Tokenizer.encode("👀👀👀"));  // 9468, 239, 222, 9468, 239, 222, 9468, 239, 222
  console.log(llama3Tokenizer.encode("👍"));  // 9468, 239, 235
  console.log(llama3Tokenizer.encode("\n😊"));  // 198, 76460, 232
  console.log(llama3Tokenizer.encode("😊"));  // 76460, 232

  console.log(llama3Tokenizer.decode(new Int32Array([9468])));  // �
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239])));  // �
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 239])));  // 👑
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 239, 9468])));  // 👑�
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 239, 9468, 239])));  // 👑�
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 239, 9468, 239, 239])));  // 👑👑

  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 239, 9468, 9468])));  // 👑��
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 239, 9468, 9468, 9468])));  // 👑���

  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 235, 9468,])));  // 👍�
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 235, 9468, 237])));  // 👍�
  console.log(llama3Tokenizer.decode(new Int32Array([9468, 239, 235, 9468, 237, 121])));  // 👍🏽

  // const decoded_para = llama3Tokenizer.decode(encoded_para);
  // console.log(decoded_para);

  const tokenizer = await loadTokenizer(useJSON, useLlama3);
  const para_input_tokens = useLlama3 ? encoded_para : llama2_para_input_tokens;

  const textStreamer = new TextStreamer(tokenizer);
  let total_text = "";
  for (const token of para_input_tokens) {
    total_text += textStreamer.put([token]);
  }
  total_text += textStreamer.finish();
  if (total_text !== DECODED_PARAGRAPH) {
    console.log("FAIL: test_text_streamer, useJSON=" + useJSON + ", useLlama3=" + useLlama3);
    throw Error("Got " + total_text + "while expect one of: " + DECODED_PARAGRAPH);
  }
  console.log("Passed test_text_streamer, useJSON=" + useJSON + ", useLlama3=" + useLlama3);
}

// Llama 3 does not have tokenizer.model
test_text_streamer_emojis(/*useJSON=*/true, /*useLlama3=*/true);
test_text_streamer_emojis(/*useJSON=*/true, /*useLlama3=*/false);
test_text_streamer_emojis(/*useJSON=*/true, /*useLlama3=*/false);

test_text_streamer(/*useJSON=*/true, /*useLlama3=*/true);
test_text_streamer(/*useJSON=*/true, /*useLlama3=*/false);
test_text_streamer(/*useJSON=*/true, /*useLlama3=*/false);
