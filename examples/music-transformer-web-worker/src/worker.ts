// Serve the chat workload through web worker
import { ChatWorkerHandler, ChatModule, LogitProcessor, WorkerMessage, CustomRequestParams, ChatInterface } from "@mlc-ai/web-llm";
import { MusicLogitProcessor } from "./music_logit_processor";
import { chunkGenerator } from "./music_transformer_generate";


const musicLogitProcessor = new MusicLogitProcessor();
const logitProcessorRegistry = new Map<string, LogitProcessor>();

logitProcessorRegistry.set("music-medium-800k-q0f32", musicLogitProcessor);

class CustomChatWorkerHandler extends ChatWorkerHandler {
  private chunkGenerator: AsyncGenerator<Array<number>, void, void>

  constructor(chat: ChatInterface) {
    super(chat);
    this.chunkGenerator = chunkGenerator(chat, musicLogitProcessor);
  }

  onmessage(event: MessageEvent<any>): void {
    const msg = event.data as WorkerMessage;
    switch (msg.kind) {
      case "customRequest": {
        const params = msg.content as CustomRequestParams;
        if (params.requestName == 'chunkGenerate') {
          console.log("Worker: generating music-transformer tokens...");
          this.handleTask(msg.uuid, async () => {
            const { value } = await this.chunkGenerator.next();
            console.log("Worker: done generating");
            return value!;
          });
        } else if (params.requestName == "resetGenerator") {
          console.log("Worker: reset music-transformer generator");
          this.handleTask(msg.uuid, async () => {
            musicLogitProcessor.resetState();
            this.chunkGenerator = chunkGenerator(chat, musicLogitProcessor);
            return null;
          })
        }
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
