// File upload handling
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');
let selectedFile = null;

// Click to upload
dropZone.addEventListener('click', () => {
    fileInput.click();
});

// File selection
fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

// Drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file) return;
    
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        alert('File size exceeds 50MB limit');
        return;
    }
    
    selectedFile = file;
    analyzeBtn.disabled = false;
    
    // Update drop zone text
    const dropText = dropZone.querySelector('.drop-text');
    dropText.textContent = `Selected: ${file.name}`;
    dropText.style.color = '#00d9ff';
}

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Show success message
            analyzeBtn.textContent = '✓ Analysis Complete';
            analyzeBtn.style.background = '#00ff88';
            
            // Store result in sessionStorage
            sessionStorage.setItem('analysisResult', JSON.stringify({
                filename: selectedFile.name,
                filesize: selectedFile.size,
                result: result,
                timestamp: Date.now()
            }));
            
            // Small delay for visual feedback
            setTimeout(() => {
                // Redirect to results page with loading indicator
                window.location.href = 'results.html?id=' + Date.now() + '&loading=true';
            }, 500);
        } else {
            alert('Analysis failed. Please try again.');
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Start Analysis';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error. Please check if the API server is running (python api_server.py)');
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Start Analysis';
    }
});

// Smooth scroll to upload section
function scrollToUpload() {
    document.getElementById('uploadSection').scrollIntoView({ 
        behavior: 'smooth',
        block: 'center'
    });
}

// Auto-detect badge animation
const badge = document.querySelector('.badge');
let hue = 180;
setInterval(() => {
    hue = (hue + 1) % 360;
    badge.style.borderColor = `hsla(${hue}, 100%, 50%, 0.3)`;
}, 50);
