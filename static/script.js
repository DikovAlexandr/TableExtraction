document.getElementById("file-input").onchange = function() {
    document.querySelector(".custom-file-upload").innerHTML = this.files[0].name;
};

function showLoadingAnimation() {
    document.getElementById('loading-animation').style.display = 'block';
}

function hideLoadingAnimation() {
    document.getElementById('loading-animation').style.display = 'none';
}