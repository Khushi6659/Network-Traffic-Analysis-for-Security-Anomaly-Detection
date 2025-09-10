// Show filename after selecting a file
document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.querySelector("#file-input");
    const fileNameDisplay = document.querySelector("#file-name");

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                fileNameDisplay.textContent = "Selected: " + fileInput.files[0].name;
            }
        });
    }
});

document.addEventListener("DOMContentLoaded", () => {
    const themeBtn = document.getElementById("themeBtn");
    const body = document.body;

    themeBtn.addEventListener("click", () => {
        body.classList.toggle("dark-theme");
        body.classList.toggle("light-theme");

        if (body.classList.contains("dark-theme")) {
            themeBtn.textContent = "☀️ Light Mode";
        } else {
            themeBtn.textContent = "🌙 Dark Mode";
        }
    });
});
