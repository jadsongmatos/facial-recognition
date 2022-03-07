var model;
const classifier = knnClassifier.create();

async function app() {
  // Load the model.
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log("Successfully loaded model");

  let start = window.performance.now();
  let end;

  classifier.addExample(
    model.infer(document.getElementById("angelina_jolie"), true),
    "angelina_jolie"
  );

  end = window.performance.now();
  console.log("angelina_jolie", end - start);
  start = window.performance.now();

  classifier.addExample(
    model.infer(document.getElementById("bill_gates"), true),
    "bill_gates"
  );

  end = window.performance.now();
  console.log("bill_gates", end - start);
  start = window.performance.now();

  classifier.addExample(
    model.infer(document.getElementById("neymar"), true),
    "neymar"
  );

  end = window.performance.now();
  console.log("neymar", end - start);
  start = window.performance.now();

  // Get the most likely class and confidence from the classifier module.
  const infer = model.infer(document.getElementById("lenna"), "conv_preds");

  const model_predict_input = await classifier.predictClass(infer, 16);
  console.log(model_predict_input);

  let result_input = 0;
  let max = 1;
  Object.values(model_predict_input.confidences).forEach((categories) => {
    result_input = result_input + categories;
    if (categories > max) {
      max = categories;
    }
  });

  console.log("result:", result_input / max, "max:", max);
}

app();
