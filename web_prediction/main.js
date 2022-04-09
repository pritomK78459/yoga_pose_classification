
function PreviewImage() {
    var oFReader = new FileReader();
    oFReader.readAsDataURL(document.getElementById("image-selector").files[0]);

    oFReader.onload = function (oFREvent) {
        document.getElementById("selected-image").src = oFREvent.target.result;
        $("#prediction-list").empty();
    };
};

async function getPred(){
    const model = await tf.loadLayersModel('http://127.0.0.1:8887/model.json');
    let image = $("#selected-image").get(0);
    image = tf.browser.fromPixels(image).toFloat().resizeNearestNeighbor([224,224]);
    image = image.expandDims();
    console.log("Image Shape",image.shape);
    predictions = await model.predict(image).data();
    console.log(predictions);

    predictions.forEach(function (p){

        var predicted_pose = predictions.indexOf(p);
        var confidence = p

        if ( predicted_pose == 0 ) {
            var className = "Downdog"
        } else if (predicted_pose == 1) {
            var className = "Goddess"
        }
        else if (predicted_pose == 2) {
            var className = "Plank"
        }
        else if (predicted_pose == 3) {
            var className = "Tree"
        }
        else if (predicted_pose == 4) {
            var className = "Warrior"
        }

        
        $('#prediction-list').append('<li>'+className+' : '+confidence+'</li>')
    });

}

