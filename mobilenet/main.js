var model;
const classifier = knnClassifier.create();

async function app() {
  // Load the model.
  model = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log("Successfully loaded model");

  const imgEu = document.getElementById("eu");
  const activation = model.infer(imgEu, true);
  //console.log("eu activation", await activation.array());
  classifier.addExample(activation, 0);

  const inferBill = model.infer(document.getElementById("bill"), true);
  //console.log("bill activation",await inferBill.array() )
  classifier.addExample(inferBill, 1);

  const inferBill2 = model.infer(document.getElementById("bill2"), true);
  classifier.addExample(inferBill2, 1);

  const inferBill3 = model.infer(document.getElementById("bill3"), true);
  classifier.addExample(inferBill3, 1);

  const inferEu2 = model.infer(document.getElementById("eu2"), true);
  classifier.addExample(inferEu2, 0);

  const inferEu3 = model.infer(document.getElementById("eu3"), "conv_preds");
  // Get the most likely class and confidence from the classifier module.
  const result = await classifier.predictClass(inferEu3);
  console.log(result);
  //console.log("eu2 activation", await inferEu2.array())
}

app();
