import * as webllm from "@mlc-ai/web-llm";
import { MusicLogitProcessor } from "./music_logit_processor";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

class CustomChatWorkerClient extends webllm.ChatWorkerClient {
  constructor(worker: any) {
    super(worker);
  }

  async chunkGenerate(): Promise<Array<number>> {
    const msg: webllm.WorkerMessage = {
      kind: "customRequest",
      uuid: crypto.randomUUID(),
      content: {
        requestName: "chunkGenerate",
        requestMessage: ""
      }
    };
    return await this.getPromise<Array<number>>(msg);
  }

  async resetGenerator(): Promise<void> {
    const msg: webllm.WorkerMessage = {
      kind: "customRequest",
      uuid: crypto.randomUUID(),
      content: {
        requestName: "resetGenerator",
        requestMessage: ""
      }
    };
    await this.getPromise<null>(msg);
  }
}

async function main() {
  const musicLogitProcessor = new MusicLogitProcessor();
  const logitProcessorRegistry = new Map<string, webllm.LogitProcessor>();
  logitProcessorRegistry.set("music-medium-800k-q0f32", musicLogitProcessor);
  const chat = new CustomChatWorkerClient(new Worker(
    new URL('./worker.ts', import.meta.url),
    { type: 'module' }
  ));

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  // Define modelRecord
  const myAppConfig: webllm.AppConfig = {
    model_list: [
      {
        "model_url": "https://huggingface.co/mlc-ai/mlc-chat-stanford-crfm-music-medium-800k-q0f32/resolve/main/",
        "local_id": "music-medium-800k-q0f32",
        "model_lib_url": "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/music-medium-800k-q0f32.wasm",
      },
    ]
  }

  // Reload chat module with a logit processor
  await chat.reload("music-medium-800k-q0f32", undefined, myAppConfig);

  let generationStopped = true;

  const startButton = document.getElementById("startButton");
  const pauseButton = document.getElementById("pauseButton");
  const resetButton = document.getElementById("resetButton");

  if (startButton) {
    startButton.addEventListener("click", async () => {
      generationStopped = false;

      while (!generationStopped) {
        const tokens = await chat.chunkGenerate();
        console.log("UI: received generated tokens: ");
        console.log(tokens);
        console.log(await chat.runtimeStatsText());
      }
    });
  }

  if (pauseButton) {
    pauseButton.addEventListener("click", async () => {
      generationStopped = true;
      console.log("UI: Pausing generator");
    });
  }

  if (resetButton) {
    resetButton.addEventListener("click", async () => {
      generationStopped = true;
      console.log("UI: Reset generator");
      await chat.resetChat();
      await chat.resetGenerator();
    });
  }
}

main();
