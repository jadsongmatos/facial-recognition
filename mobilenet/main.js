var model;
const classifier = knnClassifier.create();

async function app() {
  // Load the model.
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log("Successfully loaded model");

  const inferEu = model.infer(document.getElementById("eu"), true);
  classifier.addExample(inferEu, 0);

  const inferEu2 = model.infer(document.getElementById("eu2"), true);
  classifier.addExample(inferEu2, 0);

  const inferEu3 = model.infer(document.getElementById("eu3"), true);
  classifier.addExample(inferEu3, 0);

  const inferBill = model.infer(document.getElementById("bill"), true);
  classifier.addExample(inferBill, 1);

  const inferBill2 = model.infer(document.getElementById("bill2"), true);
  classifier.addExample(inferBill2, 1);

  const inferBill3 = model.infer(document.getElementById("bill3"), true);
  classifier.addExample(inferBill3, 1);

  // Get the most likely class and confidence from the classifier module.
  const inferEu4 = model.infer(document.getElementById("eu4"), "conv_preds");
  const model_predict_input = await classifier.predictClass(inferEu4);
  console.log(model_predict_input)
  let result_input = 0;
  let max = 1;
  Object.values(model_predict_input.confidences).forEach((categories) => {
    result_input = result_input + categories;
    if (categories > max) {
      max = categories;
    }
  });
  console.log("result:",result_input / max,"max:",max);
}

app();
