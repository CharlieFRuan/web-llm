// Serve the chat workload through web worker
import { ChatWorkerHandler, ChatModule, LogitProcessor, WorkerMessage, CustomRequestParams, ChatInterface } from "@mlc-ai/web-llm";
import { MusicLogitProcessor } from "./music_logit_processor";
import { chunkGenerator } from "./music_transformer_generate";


const musicLogitProcessor = new MusicLogitProcessor();
const logitProcessorRegistry = new Map<string, LogitProcessor>();

logitProcessorRegistry.set("music-medium-800k-q0f32", musicLogitProcessor);

class CustomChatWorkerHandler extends ChatWorkerHandler {
  constructor(chat: ChatInterface) {
    super(chat);
  }

  onmessage(event: MessageEvent<any>): void {
    const msg = event.data as WorkerMessage;
    switch (msg.kind) {
      case "customRequest": {
        console.log("Generating music-transformer tokens...");
        super.handleTask(msg.uuid, async () => {
          const params = msg.content as CustomRequestParams;
          if (params.requestName == 'chunkGenerate') {
            for await (const nextChunk of chunkGenerator(chat, musicLogitProcessor)) {
              console.log(nextChunk);
            };
          }
          return null;
        })
      }
      default:
        super.onmessage(event);
    }
  }
}

const chat = new ChatModule(logitProcessorRegistry);
const handler = new CustomChatWorkerHandler(chat);
self.onmessage = (msg: MessageEvent) => {
  handler.onmessage(msg);
};
