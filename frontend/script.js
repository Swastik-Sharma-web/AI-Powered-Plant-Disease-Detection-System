document.addEventListener('DOMContentLoaded', () => {
    // 1. Navbar Scroll Effect
    const nav = document.querySelector('nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    });

    // 2. Smooth Scrolling for Anchor Links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                const navHeight = nav.offsetHeight;
                window.scrollTo({
                    top: targetElement.offsetTop - navHeight,
                    behavior: 'smooth'
                });
            }
        });
    });

    // 3. Scroll Reveal Animations (Optional addition for luxury feel)
    const fadeElements = document.querySelectorAll('.glass-card, .workflow-item, .feature-box');
    const observerOptions = {
        threshold: 0.1,
        rootMargin: "0px 0px -50px 0px"
    };

    const fadeObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    fadeElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        fadeObserver.observe(el);
    });

    // 4. Image Upload & Mock AI Detection Logic
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const uploadText = document.querySelector('.upload-text');
    const uploadSubtext = document.querySelector('.upload-subtext');
    const uploadIcon = document.querySelector('.upload-icon');
    
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultPanel = document.getElementById('resultPanel');
    
    // Result elements
    const resStatusText = document.getElementById('resStatusText');
    const resDisease = document.getElementById('resDisease');
    const resConfidence = document.getElementById('resConfidence');
    const resConfidenceBar = document.getElementById('resConfidenceBar');
    const resAdvice = document.getElementById('resAdvice');

    // Handle Drag & Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => {
            uploadArea.classList.remove('dragover');
        }, false);
    });

    // Handle File Drop
    uploadArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }, false);

    // Handle Click to Upload
    uploadArea.addEventListener('click', () => {
        // Prevent click if currently loading
        if (loadingSpinner.style.display === 'block') return;
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        // Basic image validation
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Display Preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            previewImage.src = reader.result;
            showPreview();
            simulateAIModelPrediction(file);
        }
    }

    function showPreview() {
        uploadIcon.style.display = 'none';
        uploadText.style.display = 'none';
        uploadSubtext.style.display = 'none';
        previewContainer.style.display = 'block';
        resultPanel.style.display = 'none'; // Hide old results
    }

    // placeholder API endpoint config designed for simple replacement
    const API_ENDPOINT = '/api/predict'; 

    function simulateAIModelPrediction(file) {
        // Show loading state
        loadingSpinner.style.display = 'block';
        
        // --- REAL BACKEND CONNECTION (FastAPI) ---
        const formData = new FormData();
        formData.append('file', file);
        
        // Pointing to Hugging Face cloud server
        fetch('https://swastik1333-plant-disease-detection-system.hf.space/api/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error connecting to prediction server.\n\nThe AI backend might still be loading into memory (it takes 15-30 seconds on startup), or it may have crashed.\n\nPlease check the black terminal window for errors, and try uploading again in a few seconds!');
            loadingSpinner.style.display = 'none';
        });
    }

    function displayResults(data) {
        loadingSpinner.style.display = 'none';
        resultPanel.style.display = 'block';

        // Update DOM
        if (data.status.toLowerCase() === 'healthy') {
            resStatusText.className = 'status-badge status-healthy';
            resStatusText.innerHTML = '<i class="fa-solid fa-check-circle"></i> Healthy';
        } else {
            resStatusText.className = 'status-badge status-diseased';
            resStatusText.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Diseased';
        }

        resDisease.textContent = data.diseaseName;
        resConfidence.textContent = data.confidencePercent;
        
        // Timeout needed to trigger CSS transition inside newly visible container
        setTimeout(() => {
            resConfidenceBar.style.width = data.confidencePercent;
        }, 50);

        resAdvice.textContent = data.advice;
    }
});
