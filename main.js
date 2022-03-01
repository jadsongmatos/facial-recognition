let net;
const classifier = knnClassifier.create();

async function app() {
  console.log("Loading mobilenet..");

  // Load the model.
  net = await mobilenet.load({ version: 2, alpha: 1.0 });
  console.log("Successfully loaded model");

  const imgEu = document.getElementById("eu");
  const activation = net.infer(imgEu, true);
  //console.log("eu activation", await activation.array());
  classifier.addExample(activation, 0);


  const bill = document.getElementById("bill");
  const inferBill = net.infer(bill, true)
  //console.log("bill activation",await inferBill.array() )
  classifier.addExample(inferBill,1);

  const eu2 = document.getElementById("eu2");
  const inferEu2 = net.infer(eu2, true)
  classifier.addExample(inferEu2,0);

  const eu3 = document.getElementById("eu3");
  const inferEu3 = net.infer(eu3, 'conv_preds');
  // Get the most likely class and confidence from the classifier module.
  const result = await classifier.predictClass(inferEu3);
  console.log(result)
  //console.log("eu2 activation", await inferEu2.array())
}

app();
