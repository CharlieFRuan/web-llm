import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

async function main() {
  const chat = new webllm.ChatModule();

  chat.setInitProgressCallback((report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  });

  await chat.reload("music-medium-800k-q0f32");

  const generateProgressCallback = (_step: number, message: string) => {
    setLabel("generate-label", message);
  };

  // Get next logits
  // const firstToken: Array<number> = [55026];
  // setLabel("prompt-label", firstToken.toString());
  // const logits = await chat.forwardTokens(firstToken, 1, isPrefill=true);
  // console.log(logits);

  // Get next token
  const prompt: Array<number> = [55026];
  setLabel("prompt-label", prompt.toString());
  let nextToken = await chat.forwardTokensAndSample(prompt, prompt.length, isPrefill = true);
  console.log(nextToken);

  let counter = prompt.length;
  while (counter < 64) {
    counter += 1;
    nextToken = await chat.forwardTokensAndSample([nextToken], counter, isPrefill = false);
    console.log(nextToken);
  }

  console.log(await chat.runtimeStatsText());
}

main();
