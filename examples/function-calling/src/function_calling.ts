/**
 * This example follows https://platform.openai.com/docs/guides/function-calling,
 * where we tell the model about the `getCurrentWeather()` function, ask it to generate
 * the function call, execute the function call, tell the model the function call's result,
 * and let it summarize the results.
 */

import * as webllm from "@mlc-ai/web-llm";

function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label == null) {
    throw Error("Cannot find label " + id);
  }
  label.innerText = text;
}

/**
 * Dummy function that can get called with the output of the model's tool_calls output.
 */
function getCurrentWeather(location: string, unit: string): number {
  if (location === "Pittsburgh") {
    if (unit === "celsius") {
      return 23;
    } else if (unit === "fahrenheit") {
      return 73.4;
    } else {
      throw new Error("Unexpected unit: " + unit + "please rerun example.");
    }
  } else if (location === "Tokyo") {
    if (unit === "celsius") {
      return 18;
    } else if (unit === "fahrenheit") {
      return 66.2;
    } else {
      throw new Error("Unexpected unit: " + unit + "please rerun example.");
    }
  } else {
    throw new Error(
      "Unexpected location: " + location + "please rerun example.",
    );
  }
}

async function main() {
  // Step 0. Load the model
  const initProgressCallback = (report: webllm.InitProgressReport) => {
    setLabel("init-label", report.text);
  };
  const selectedModel = "Hermes-2-Pro-Llama-3-8B-q4f16_1-MLC";
  const engine: webllm.MLCEngineInterface = await webllm.CreateMLCEngine(
    selectedModel,
    { initProgressCallback: initProgressCallback },
  );

  // Step 1: send the conversation and available functions to the model
  const tools: Array<webllm.ChatCompletionTool> = [
    {
      type: "function",
      function: {
        name: "get_current_weather",
        description: "Get the current weather in a given location",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city and state, e.g. San Francisco, CA",
            },
            unit: { type: "string", enum: ["celsius", "fahrenheit"] },
          },
          required: ["location"],
        },
      },
    },
  ];

  const messages = [
    {
      role: "user",
      content: "What is the current weather in celsius in Pittsburgh?",
    },
  ] as Array<webllm.ChatCompletionMessageParam>;

  const request: webllm.ChatCompletionRequest = {
    stream: false, // works with stream as well, where the last chunk returns tool_calls
    messages: messages,
    tool_choice: "auto",
    tools: tools,
  };

  const response = await engine.chat.completions.create(request);
  const responseMessage = response.choices[0].message;
  console.log("responseMessage: ", responseMessage);

  // Step 2: check if the model wanted to call a function
  const toolCalls = responseMessage.tool_calls;
  if (toolCalls) {
    // Step 3: call the function
    // Note: the JSON response is guaranteed to be in JSON format, but not always match
    // the function header, so be sure to catch error and retry if needed
    const availableFunctions = {
      get_current_weather: getCurrentWeather,
    }; // only one function in this example, but you can have multiple
    messages.push(responseMessage); // extend conversation with assistant's reply
    for (const toolCall of toolCalls) {
      const functionName = toolCall.function.name;
      const functionToCall = availableFunctions[functionName];
      const functionArgs = JSON.parse(toolCall.function.arguments);
      const functionResponse = functionToCall(
        functionArgs.location,
        functionArgs.unit,
      );
      messages.push({
        tool_call_id: toolCall.id,
        role: "tool",
        content: functionResponse,
      }); // extend conversation with function response
    }
    const secondResponse = await engine.chat.completions.create({
      messages: messages,
    }); // get a new response from the model where it can see the function response
    return secondResponse.choices;
  }

  console.log(await engine.runtimeStatsText());
}

main();
