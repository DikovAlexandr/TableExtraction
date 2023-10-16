document.getElementById("file-input").onchange = function() {
    document.querySelector(".custom-file-upload").innerHTML = this.files[0].name;
};