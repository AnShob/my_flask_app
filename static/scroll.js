// static/js/scroll.js

// Fungsi untuk menghapus teks placeholder saat textarea mendapatkan focus
function clearText() {
    var textarea = document.getElementById("Sinopsis");
    if (textarea.value === "Ketik sinopsis di sini...") {
        textarea.value = "";
    }
}

// Fungsi untuk mengembalikan teks placeholder jika textarea kosong
function restoreText() {
    var textarea = document.getElementById("Sinopsis");
    if (textarea.value === "") {
        textarea.value = "Ketik sinopsis di sini...";
    }
}

// Fungsi untuk scroll otomatis ke bagian hasil setelah form disubmit
function scrollToResults() {
    setTimeout(function() {
        var resultSection = document.getElementById("genre-result");
        if (resultSection) {
            resultSection.scrollIntoView({ behavior: 'smooth' });
        }
    }, 1000); // 1000ms delay untuk menunggu halaman selesai render
    // Menambahkan hash pada URL agar halaman menggulir ke genre-result
function addHashToURL() {
    window.location.hash = 'genre-result';
}
}
