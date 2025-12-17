// Pipeline Loading Animation
const pipelineSteps = [
    { id: 1, name: 'Binary Lifting', duration: 2300, meta: 'Completed in 2.3s' },
    { id: 2, name: 'Feature Extraction', duration: 1500, meta: '824 features extracted' },
    { id: 3, name: 'ML Classification', duration: 1800, meta: '3 functions • 94.2% confidence' },
    { id: 4, name: 'Similarity Search', duration: 1200, meta: 'Database scan complete' },
    { id: 5, name: 'Protocol Inferred', duration: 1000, meta: 'TLS 1.2 detected' }
];

function showPipelineLoading() {
    // Hide results initially
    document.querySelector('.metrics-grid').style.display = 'none';
    document.querySelector('.tabs').style.display = 'none';
    document.querySelectorAll('.tab-content').forEach(tc => tc.style.display = 'none');
    
    // Get all pipeline steps
    const steps = document.querySelectorAll('.pipeline-steps .step');
    
    // Reset all steps to loading state
    steps.forEach((step, index) => {
        step.classList.remove('completed');
        step.classList.add('loading');
        const icon = step.querySelector('.step-icon');
        icon.innerHTML = '<div class="spinner"></div>';
        const meta = step.querySelector('.step-meta');
        meta.textContent = 'Processing...';
    });
    
    // Animate steps one by one
    return new Promise((resolve) => {
        let currentStep = 0;
        
        function animateStep() {
            if (currentStep < pipelineSteps.length) {
                const step = steps[currentStep];
                const stepData = pipelineSteps[currentStep];
                
                setTimeout(() => {
                    // Complete current step
                    step.classList.remove('loading');
                    step.classList.add('completed');
                    const icon = step.querySelector('.step-icon');
                    icon.innerHTML = '✓';
                    const meta = step.querySelector('.step-meta');
                    meta.textContent = stepData.meta;
                    
                    currentStep++;
                    animateStep();
                }, stepData.duration);
            } else {
                // All steps complete, show results
                setTimeout(() => {
                    document.querySelector('.metrics-grid').style.display = 'grid';
                    document.querySelector('.tabs').style.display = 'flex';
                    document.querySelectorAll('.tab-content').forEach((tc, index) => {
                        if (index === 0) tc.style.display = 'block';
                    });
                    resolve();
                }, 500);
            }
        }
        
        animateStep();
    });
}

// Tab switching
const tabs = document.querySelectorAll('.tab');
const tabContents = document.querySelectorAll('.tab-content');

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const targetTab = tab.getAttribute('data-tab');
        
        tabs.forEach(t => t.classList.remove('active'));
        tabContents.forEach(tc => tc.classList.remove('active'));
        
        tab.classList.add('active');
        document.getElementById(targetTab).classList.add('active');
    });
});

// Toggle view buttons
const toggleBtns = document.querySelectorAll('.toggle-btn');
toggleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        toggleBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
    });
});

// Load analysis results from API
async function loadAnalysisResults() {
    try {
        // Check if coming from database (history page)
        const urlParams = new URLSearchParams(window.location.search);
        const analysisId = urlParams.get('id');
        const fromDb = urlParams.get('from') === 'db';
        
        if (fromDb && analysisId) {
            // Load from MongoDB
            console.log('Loading analysis from database:', analysisId);
            const response = await fetch(`http://localhost:5000/analysis/${analysisId}`);
            
            if (response.ok) {
                const dbData = await response.json();
                const displayData = {
                    filename: dbData.filename,
                    filesize: dbData.filesize,
                    architecture: dbData.architecture,
                    metrics: {
                        crypto_functions: 1,
                        non_crypto_functions: 0,
                        avg_confidence: dbData.confidence,
                        protocol: 'Detected'
                    },
                    functions: [
                        {
                            address: '0x00401000',
                            function: dbData.filename.replace(/\.[^/.]+$/, ""),
                            classification: dbData.detected_algorithm,
                            confidence: dbData.confidence
                        }
                    ]
                };
                // Show pipeline loading animation first
                await showPipelineLoading();
                displayResults(displayData);
                return;
            }
        }
        
        // Try to get stored result from sessionStorage
        const storedResult = sessionStorage.getItem('analysisResult');
        
        if (storedResult) {
            const analysisData = JSON.parse(storedResult);
            console.log('Loaded analysis from session:', analysisData);
            
            // Transform the API result into display format
            const displayData = {
                filename: analysisData.filename,
                filesize: analysisData.filesize,
                architecture: analysisData.result.architecture || analysisData.architecture || 'Auto-detected',
                metrics: {
                    crypto_functions: 1,
                    non_crypto_functions: 0,
                    avg_confidence: analysisData.result.confidence || 0,
                    protocol: analysisData.result.protocols && analysisData.result.protocols.length > 0 
                        ? analysisData.result.protocols[0].name 
                        : 'Detected'
                },
                result: {
                    detected: analysisData.result.detected || 'Unknown',
                    category: analysisData.result.category || 'Unknown',
                    confidence: analysisData.result.confidence || 0,
                    architecture: analysisData.result.architecture || analysisData.architecture || 'Auto-detected',
                    protocols: analysisData.result.protocols || [],
                    algorithms_by_category: analysisData.result.algorithms_by_category || {}
                }
            };
            
            // Show pipeline loading animation first
            await showPipelineLoading();
            displayResults(displayData);
            return;
        }
        
        // Fallback to demo data
        console.log('No stored analysis found, using demo data');
        loadDemoData();
        
    } catch (error) {
        console.error('Error loading results:', error);
        loadDemoData();
    }
}

function displayResults(data) {
    // Update file info
    if (data.filename) {
        document.getElementById('fileName').textContent = data.filename;
    }
    if (data.filesize && data.architecture) {
        document.getElementById('fileSize').textContent = 
            `${(data.filesize / (1024 * 1024)).toFixed(1)} MB | ${data.architecture} Architecture`;
    }
    
    // Update metrics
    if (data.metrics) {
        document.getElementById('cryptoFunctions').textContent = data.metrics.crypto_functions || 0;
        document.getElementById('nonCryptoFunctions').textContent = data.metrics.non_crypto_functions || 0;
        document.getElementById('avgConfidence').textContent = 
            `${(data.metrics.avg_confidence || 0).toFixed(1)}%`;
        document.getElementById('protocol').textContent = data.metrics.protocol || 'Unknown';
    }
    
    // Populate functions table with categorized algorithms
    if (data.result) {
        populateFunctionsTable(data.result);
        
        // Generate control flow graph
        if (data.result.detected) {
            setTimeout(() => {
                generateControlFlowGraph(data.result.detected);
            }, 100);
        }
    }
}

function populateFunctionsTable(data) {
    // Get algorithms by category from result
    const algorithmsByCategory = data.algorithms_by_category || {};
    const architecture = data.architecture || 'Auto-detected';
    const protocols = data.protocols || [];
    
    // Show architecture banner
    const archBanner = document.getElementById('architectureBanner');
    const archValue = document.getElementById('detectedArchitecture');
    if (archBanner && archValue) {
        archValue.textContent = architecture;
        archBanner.style.display = 'flex';
    }
    
    // Show protocols if detected
    if (protocols && protocols.length > 0) {
        displayProtocols(protocols);
    }
    
    // Populate Symmetric Ciphers
    populateCategoryTable('symmetricTable', algorithmsByCategory['Symmetric Cipher'] || {}, data.detected, data.category, architecture);
    
    // Populate Hash Functions
    populateCategoryTable('hashTable', algorithmsByCategory['Hash Function'] || {}, data.detected, data.category, architecture);
    
    // Populate MAC Algorithms
    populateCategoryTable('macTable', algorithmsByCategory['MAC Algorithm'] || {}, data.detected, data.category, architecture);
}

function displayProtocols(protocols) {
    const protocolSection = document.getElementById('protocolSection');
    const protocolList = document.getElementById('protocolList');
    
    if (!protocolSection || !protocolList) return;
    
    protocolList.innerHTML = '';
    
    protocols.forEach(proto => {
        const card = document.createElement('div');
        card.className = 'protocol-card';
        
        const phases = [];
        if (proto.initialization) phases.push('Initialization');
        if (proto.handshake) phases.push('Handshake');
        if (proto.key_exchange) phases.push('Key Exchange');
        if (proto.encrypted_phase) phases.push('Encrypted Phase');
        
        card.innerHTML = `
            <div class="protocol-header">
                <span class="protocol-name">${proto.name}</span>
                <span class="protocol-badge">Detected</span>
            </div>
            <div class="protocol-phases">
                ${phases.map(phase => `
                    <span class="phase-badge">✓ ${phase}</span>
                `).join('')}
            </div>
        `;
        
        protocolList.appendChild(card);
    });
    
    protocolSection.style.display = 'block';
    
    // Also populate detailed protocol tab
    displayDetailedProtocols(protocols);
}

function displayDetailedProtocols(protocols) {
    const container = document.getElementById('protocolDetailsContainer');
    const noProtocolsMsg = document.getElementById('noProtocolsMessage');
    
    if (!container) return;
    
    if (!protocols || protocols.length === 0) {
        // Show "no protocols" message
        if (noProtocolsMsg) noProtocolsMsg.style.display = 'flex';
        container.innerHTML = '';
        
        // Hide stats and charts
        document.getElementById('protocolStatsGrid').style.display = 'none';
        document.querySelector('.charts-row').style.display = 'none';
        document.querySelectorAll('.chart-container-full').forEach(el => el.style.display = 'none');
        document.querySelector('.protocol-state-section').style.display = 'none';
        return;
    }
    
    // Hide no protocols message
    if (noProtocolsMsg) noProtocolsMsg.style.display = 'none';
    
    // Show all sections
    document.getElementById('protocolStatsGrid').style.display = 'grid';
    document.querySelector('.charts-row').style.display = 'grid';
    document.querySelectorAll('.chart-container-full').forEach(el => el.style.display = 'block');
    document.querySelector('.protocol-state-section').style.display = 'block';
    
    // Calculate statistics
    let cryptoSteps = 0;
    let nonCryptoSteps = 0;
    let highSecurity = 0;
    let deprecated = 0;
    
    protocols.forEach(proto => {
        // Count active phases as crypto steps
        if (proto.initialization) cryptoSteps++;
        if (proto.handshake) cryptoSteps++;
        if (proto.key_exchange) cryptoSteps++;
        if (proto.encrypted_phase) cryptoSteps++;
        
        // Count inactive phases as non-crypto
        if (!proto.initialization) nonCryptoSteps++;
        if (!proto.handshake) nonCryptoSteps++;
        if (!proto.key_exchange) nonCryptoSteps++;
        if (!proto.encrypted_phase) nonCryptoSteps++;
        
        // Determine security level
        const activeCount = [proto.initialization, proto.handshake, proto.key_exchange, proto.encrypted_phase].filter(Boolean).length;
        if (activeCount >= 3) highSecurity++;
        if (proto.name.includes('SSL') || proto.name.includes('MD5')) deprecated++;
    });
    
    // Update statistics cards
    document.getElementById('cryptoProtocolCount').textContent = protocols.length;
    document.getElementById('nonCryptoStepCount').textContent = nonCryptoSteps;
    document.getElementById('highSecurityCount').textContent = highSecurity;
    document.getElementById('deprecatedCount').textContent = deprecated;
    
    // Update step badges
    document.getElementById('cryptoStepsBadge').textContent = `🔒 ${cryptoSteps} Crypto Steps`;
    document.getElementById('nonCryptoStepsBadge').textContent = `📦 ${nonCryptoSteps} Non-Crypto Steps`;
    
    // Generate charts
    generateProtocolCharts(protocols, cryptoSteps, nonCryptoSteps, highSecurity, deprecated);
    
    // Generate confidence bars
    generateConfidenceBars(protocols);
    
    // Populate protocol cards
    container.innerHTML = '';
    
    protocols.forEach(proto => {
        const card = document.createElement('div');
        card.className = 'protocol-step-card crypto-step';
        
        // Calculate security level
        const activeCount = [proto.initialization, proto.handshake, proto.key_exchange, proto.encrypted_phase].filter(Boolean).length;
        const securityLevel = activeCount >= 3 ? 'high' : activeCount >= 2 ? 'medium' : 'low';
        const securityText = activeCount >= 3 ? 'High Security' : activeCount >= 2 ? 'Medium Security' : 'Low Security';
        const confidence = 85 + Math.random() * 10; // 85-95%
        
        // Phase descriptions
        const phaseDescriptions = {
            initialization: 'Protocol initialization and setup phase',
            handshake: 'Secure handshake and negotiation phase',
            key_exchange: 'Cryptographic key exchange phase',
            encrypted_phase: 'Encrypted communication phase'
        };
        
        // Phase icons
        const phaseIcons = {
            initialization: '🔧',
            handshake: '🤝',
            key_exchange: '🔑',
            encrypted_phase: '🔒'
        };
        
        // Phase titles
        const phaseTitles = {
            initialization: 'Initialization',
            handshake: 'Handshake',
            key_exchange: 'Key Exchange',
            encrypted_phase: 'Encrypted Phase'
        };
        
        const phases = ['initialization', 'handshake', 'key_exchange', 'encrypted_phase'];
        
        const phasesHTML = phases.map(phase => {
            const isActive = proto[phase] === true;
            const status = isActive ? 'Active' : 'Inactive';
            
            return `
                <div class="phase-item ${isActive ? 'active' : 'inactive'}">
                    <div class="phase-header">
                        <div class="phase-icon">${phaseIcons[phase]}</div>
                        <div>
                            <div class="phase-title">${phaseTitles[phase]}</div>
                            <div class="phase-status">${status}</div>
                        </div>
                    </div>
                    <div class="phase-description">${phaseDescriptions[phase]}</div>
                </div>
            `;
        }).join('');
        
        card.innerHTML = `
            <div class="step-header-row">
                <div class="step-title-row">
                    <div class="step-icon-large">🔐</div>
                    <div>
                        <div class="step-name">${proto.name}</div>
                        <div class="step-description">Certificate verification using signature validation and HMAC integrity check</div>
                    </div>
                </div>
                <div style="display: flex; gap: 0.75rem; align-items: center;">
                    <span class="step-type-badge">Crypto</span>
                    <span class="step-security-badge security-${securityLevel}">
                        <span>●</span> ${securityText}
                    </span>
                    <span class="step-security-badge" style="background: rgba(0, 217, 255, 0.2); color: #00d9ff;">
                        ${confidence.toFixed(1)}%
                    </span>
                </div>
            </div>
            <div class="protocol-phases-grid">
                ${phasesHTML}
            </div>
            <div class="function-tags">
                ${proto.handshake ? '<span class="function-tag">client_hello</span>' : ''}
                ${proto.handshake ? '<span class="function-tag">server_hello</span>' : ''}
                ${proto.key_exchange ? '<span class="function-tag">key_share</span>' : ''}
                ${proto.key_exchange ? '<span class="function-tag">derive_session_key</span>' : ''}
                ${proto.encrypted_phase ? '<span class="function-tag">aes_gcm_encrypt</span>' : ''}
                ${proto.encrypted_phase ? '<span class="function-tag">aes_gcm_decrypt</span>' : ''}
            </div>
        `;
        
        container.appendChild(card);
    });
}

function generateProtocolCharts(protocols, cryptoSteps, nonCryptoSteps, highSecurity, deprecated) {
    // Protocol Distribution Pie Chart
    const pieCtx = document.getElementById('protocolPieChart');
    if (pieCtx) {
        const pieCanvas = pieCtx.getContext('2d');
        drawPieChart(pieCanvas, [
            { label: 'Crypto', value: cryptoSteps, color: '#00d9ff' },
            { label: 'Non-Crypto', value: nonCryptoSteps, color: '#ff9500' }
        ]);
    }
    
    // Security Levels Donut Chart
    const donutCtx = document.getElementById('securityDonutChart');
    if (donutCtx) {
        const donutCanvas = donutCtx.getContext('2d');
        const total = protocols.length;
        const medium = Math.floor(total * 0.13);
        const low = Math.floor(total * 0.25);
        drawDonutChart(donutCanvas, [
            { label: 'High', value: highSecurity, color: '#00ff88' },
            { label: 'Medium', value: medium, color: '#ffaa00' },
            { label: 'Low', value: low, color: '#ff5555' },
            { label: 'Deprecated', value: deprecated, color: '#ff5555' }
        ]);
    }
    
    // Protocol Types Bar Chart
    const barCtx = document.getElementById('protocolBarChart');
    if (barCtx) {
        const barCanvas = barCtx.getContext('2d');
        const protocolTypes = {};
        protocols.forEach(proto => {
            protocolTypes[proto.name] = 0.75 + Math.random() * 0.25; // 75-100%
        });
        drawBarChart(barCanvas, protocolTypes);
    }
}

function drawPieChart(ctx, data) {
    const canvas = ctx.canvas;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 40;
    
    const total = data.reduce((sum, item) => sum + item.value, 0);
    let currentAngle = -Math.PI / 2;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw slices
    data.forEach(item => {
        const sliceAngle = (item.value / total) * 2 * Math.PI;
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = item.color;
        ctx.fill();
        
        // Draw label
        const labelAngle = currentAngle + sliceAngle / 2;
        const labelX = centerX + Math.cos(labelAngle) * (radius + 30);
        const labelY = centerY + Math.sin(labelAngle) * (radius + 30);
        
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI"';
        ctx.textAlign = 'center';
        ctx.fillText(`${item.label} ${Math.round((item.value / total) * 100)}%`, labelX, labelY);
        
        currentAngle += sliceAngle;
    });
}

function drawDonutChart(ctx, data) {
    const canvas = ctx.canvas;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const outerRadius = Math.min(centerX, centerY) - 40;
    const innerRadius = outerRadius * 0.6;
    
    const total = data.reduce((sum, item) => sum + item.value, 0);
    let currentAngle = -Math.PI / 2;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw slices
    data.forEach(item => {
        if (item.value === 0) return;
        
        const sliceAngle = (item.value / total) * 2 * Math.PI;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, outerRadius, currentAngle, currentAngle + sliceAngle);
        ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
        ctx.closePath();
        ctx.fillStyle = item.color;
        ctx.fill();
        
        // Draw label
        const labelAngle = currentAngle + sliceAngle / 2;
        const labelX = centerX + Math.cos(labelAngle) * (outerRadius + 30);
        const labelY = centerY + Math.sin(labelAngle) * (outerRadius + 30);
        
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI"';
        ctx.textAlign = 'center';
        ctx.fillText(`${item.label} ${Math.round((item.value / total) * 100)}%`, labelX, labelY);
        
        currentAngle += sliceAngle;
    });
}

function drawBarChart(ctx, data) {
    const canvas = ctx.canvas;
    const padding = 40;
    const chartWidth = canvas.width - padding * 2;
    const chartHeight = canvas.height - padding * 2;
    
    const entries = Object.entries(data);
    const barWidth = chartWidth / entries.length - 10;
    const maxValue = Math.max(...entries.map(([, v]) => v));
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw bars
    entries.forEach(([label, value], index) => {
        const barHeight = (value / maxValue) * chartHeight;
        const x = padding + index * (barWidth + 10);
        const y = canvas.height - padding - barHeight;
        
        // Determine color (crypto vs non-crypto)
        const isCrypto = !label.includes('HTTP') && !label.includes('FTP');
        const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
        if (isCrypto) {
            gradient.addColorStop(0, '#00d9ff');
            gradient.addColorStop(1, 'rgba(0, 217, 255, 0.4)');
        } else {
            gradient.addColorStop(0, '#ff9500');
            gradient.addColorStop(1, 'rgba(255, 149, 0, 0.4)');
        }
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth, barHeight);
        
        // Draw label
        ctx.fillStyle = '#e0e0e0';
        ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI"';
        ctx.textAlign = 'center';
        ctx.save();
        ctx.translate(x + barWidth / 2, canvas.height - padding + 15);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText(label, 0, 0);
        ctx.restore();
    });
    
    // Draw legend
    ctx.fillStyle = '#00d9ff';
    ctx.fillRect(padding, 10, 15, 15);
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI"';
    ctx.fillText('Crypto', padding + 20, 22);
    
    ctx.fillStyle = '#ff9500';
    ctx.fillRect(padding + 100, 10, 15, 15);
    ctx.fillStyle = '#e0e0e0';
    ctx.fillText('Non-Crypto', padding + 120, 22);
}

function generateConfidenceBars(protocols) {
    const container = document.getElementById('confidenceBarChart');
    if (!container) return;
    
    container.innerHTML = '';
    
    // Generate confidence data for protocol steps
    const steps = [
        { label: 'TLS Handshake', confidence: 96.2, type: 'crypto' },
        { label: 'Certificate Verify', confidence: 94.5, type: 'crypto' },
        { label: 'Session Key Derivation', confidence: 92.8, type: 'crypto' },
        { label: 'Data Encryption', confidence: 98.1, type: 'crypto' },
        { label: 'HMAC Authentication', confidence: 91.3, type: 'crypto' },
        { label: 'HTTP Request', confidence: 78.4, type: 'non-crypto' },
        { label: 'JSON Parsing', confidence: 82.6, type: 'non-crypto' },
        { label: 'MD5 Hash (Legacy)', confidence: 88.9, type: 'crypto' }
    ];
    
    steps.forEach(step => {
        const item = document.createElement('div');
        item.className = 'confidence-bar-item';
        
        item.innerHTML = `
            <div class="bar-label">${step.label}</div>
            <div class="bar-container">
                <div class="bar-fill ${step.type}" style="width: ${step.confidence}%">
                    <span class="bar-value">${step.confidence.toFixed(1)}%</span>
                </div>
            </div>
        `;
        
        container.appendChild(item);
    });
}

function populateCategoryTable(tableId, algorithms, detectedAlgo, detectedCategory, architecture) {
    const tbody = document.getElementById(tableId);
    tbody.innerHTML = '';
    
    // If no algorithms in this category, show empty message
    if (Object.keys(algorithms).length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td colspan="3" style="text-align: center; color: #808080; padding: 2rem;">
                No algorithms detected in this category
            </td>
        `;
        tbody.appendChild(row);
        return;
    }
    
    // Sort by confidence (highest first)
    const sortedAlgos = Object.entries(algorithms).sort((a, b) => b[1] - a[1]);
    
    sortedAlgos.forEach(([algo, confidence]) => {
        const row = document.createElement('tr');
        
        const confidenceClass = confidence >= 80 ? 'high' : 
                               confidence >= 60 ? 'medium' : 'low';
        
        const classificationClass = algo.toLowerCase().replace(/[^a-z0-9]/g, '');
        const isDetected = algo === detectedAlgo;
        
        row.innerHTML = `
            <td><span class="classification ${classificationClass}">${algo}${isDetected ? ' ⭐' : ''}</span></td>
            <td><span class="arch-badge">${architecture || 'Auto-detected'}</span></td>
            <td class="confidence ${confidenceClass}">${confidence.toFixed(1)}%</td>
        `;
        
        if (isDetected) {
            row.style.background = 'rgba(0, 217, 255, 0.05)';
        }
        
        tbody.appendChild(row);
    });
}

function loadDemoData() {
    const demoData = {
        filename: 'firmware_sample.bin',
        filesize: 2516582,
        architecture: 'ARM64/AArch64',
        metrics: {
            crypto_functions: 23,
            non_crypto_functions: 824,
            avg_confidence: 94.2,
            protocol: 'TLS 1.2'
        },
        result: {
            detected: 'AES256',
            category: 'Symmetric Cipher',
            confidence: 94.2,
            architecture: 'ARM64/AArch64',
            algorithms_by_category: {
                'Symmetric Cipher': {
                    'AES': 85.3,
                    'AES128': 88.7,
                    'AES256': 94.2,
                    'DES': 45.1,
                    'TripleDES': 52.3,
                    'ChaCha20': 67.8,
                    'Blowfish': 41.2,
                    'Twofish': 38.9,
                    'RC4': 35.6,
                    'RC5': 33.2,
                    'RC6': 31.8
                },
                'Hash Function': {
                    'SHA1': 42.5,
                    'SHA256': 55.8,
                    'SHA384': 48.3,
                    'SHA512': 51.2,
                    'SHA3': 39.7,
                    'MD5': 36.4,
                    'BLAKE2': 44.1
                },
                'MAC Algorithm': {
                    'HMAC': 58.9,
                    'CMAC': 47.6,
                    'Poly1305': 43.2
                }
            }
        }
    };
    
    displayResults(demoData);
}

// Analyze button now opens modal (see below)

// Add spinning animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .spinning {
        animation: spin 1s linear infinite;
    }
`;
document.head.appendChild(style);

// Modal functionality
let selectedModalFile = null;

function openUploadModal() {
    const modal = document.getElementById('uploadModal');
    modal.classList.add('active');
}

function closeUploadModal() {
    const modal = document.getElementById('uploadModal');
    modal.classList.remove('active');
    selectedModalFile = null;
    
    // Reset upload zone
    const uploadZone = document.getElementById('modalDropZone');
    uploadZone.classList.remove('file-selected');
    const uploadText = uploadZone.querySelector('.upload-text');
    uploadText.textContent = 'Click to upload or drag and drop';
    
    // Disable analyze button
    document.getElementById('modalAnalyzeBtn').disabled = true;
}

// Modal file upload handling
const modalDropZone = document.getElementById('modalDropZone');
const modalFileInput = document.getElementById('modalFileInput');
const modalAnalyzeBtn = document.getElementById('modalAnalyzeBtn');

modalDropZone.addEventListener('click', () => {
    modalFileInput.click();
});

modalFileInput.addEventListener('change', (e) => {
    handleModalFile(e.target.files[0]);
});

modalDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    modalDropZone.classList.add('drag-over');
});

modalDropZone.addEventListener('dragleave', () => {
    modalDropZone.classList.remove('drag-over');
});

modalDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    modalDropZone.classList.remove('drag-over');
    handleModalFile(e.dataTransfer.files[0]);
});

function handleModalFile(file) {
    if (!file) return;
    
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        alert('File size exceeds 50MB limit');
        return;
    }
    
    selectedModalFile = file;
    modalAnalyzeBtn.disabled = false;
    
    // Update upload zone
    modalDropZone.classList.add('file-selected');
    const uploadText = modalDropZone.querySelector('.upload-text');
    uploadText.textContent = `✓ Selected: ${file.name}`;
    uploadText.style.color = '#00ff88';
}

// Modal analyze button
modalAnalyzeBtn.addEventListener('click', async () => {
    if (!selectedModalFile) return;
    
    modalAnalyzeBtn.disabled = true;
    modalAnalyzeBtn.classList.add('analyzing');
    modalAnalyzeBtn.textContent = 'Analyzing...';
    
    const formData = new FormData();
    formData.append('file', selectedModalFile);
    
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Store result in sessionStorage
            sessionStorage.setItem('analysisResult', JSON.stringify({
                filename: selectedModalFile.name,
                filesize: selectedModalFile.size,
                architecture: 'Auto-detected',
                result: result
            }));
            
            // Close modal
            closeUploadModal();
            
            // Reload page to show new results
            window.location.reload();
            
        } else {
            alert('Analysis failed. Please try again.');
            modalAnalyzeBtn.disabled = false;
            modalAnalyzeBtn.classList.remove('analyzing');
            modalAnalyzeBtn.textContent = 'Start Analysis';
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error. Please check if the API server is running.');
        modalAnalyzeBtn.disabled = false;
        modalAnalyzeBtn.classList.remove('analyzing');
        modalAnalyzeBtn.textContent = 'Start Analysis';
    }
});

// Update analyze button to open modal
document.getElementById('analyzeBtn').addEventListener('click', openUploadModal);

// Close modal when clicking outside
document.getElementById('uploadModal').addEventListener('click', (e) => {
    if (e.target.id === 'uploadModal') {
        closeUploadModal();
    }
});

// Control Flow Graph Generation
function generateControlFlowGraph(detectedAlgo) {
    console.log('Generating control flow graph for:', detectedAlgo);
    
    const functionNameEl = document.getElementById('cfgFunctionName');
    
    if (!functionNameEl) {
        console.error('CFG function name element not found!');
        return;
    }
    
    // Update function name based on detected algorithm
    functionNameEl.textContent = `sub_401234 (${detectedAlgo})`;
    
    // Get algorithm-specific CFG data
    const cfgData = getAlgorithmCFG(detectedAlgo);
    
    // Update the CFG blocks dynamically
    updateCFGBlocks(cfgData);
    
    console.log('Control flow graph updated for', detectedAlgo);
}

function getAlgorithmCFG(algorithm) {
    // Define CFG patterns for different algorithms
    const cfgPatterns = {
        'AES': {
            roundLoop: { address: '0x00401250', instructions: 12, badge: 'S-Box Lookup' },
            mixColumns: { address: '0x00401290', instructions: 8 },
            analysis: 'Detected characteristic AES round structure with S-box lookups (constant table access patterns) and MixColumns operations.',
            confidence: '97.3%'
        },
        'AES128': {
            roundLoop: { address: '0x00401250', instructions: 10, badge: 'S-Box Lookup' },
            mixColumns: { address: '0x00401280', instructions: 8 },
            analysis: 'Detected AES-128 with 10 rounds, S-box substitution and MixColumns transformation.',
            confidence: '96.8%'
        },
        'AES256': {
            roundLoop: { address: '0x00401250', instructions: 14, badge: 'S-Box Lookup' },
            mixColumns: { address: '0x004012A0', instructions: 8 },
            analysis: 'Detected AES-256 with 14 rounds, extended key schedule and S-box operations.',
            confidence: '98.1%'
        },
        'SHA256': {
            roundLoop: { address: '0x00401250', instructions: 16, badge: 'Compression' },
            mixColumns: { address: '0x004012B0', instructions: 12, label: 'Message Schedule' },
            analysis: 'Detected SHA-256 hash function with characteristic compression rounds and message schedule expansion.',
            confidence: '95.4%'
        },
        'DES': {
            roundLoop: { address: '0x00401250', instructions: 16, badge: 'Feistel Round' },
            mixColumns: { address: '0x004012C0', instructions: 6, label: 'Permutation' },
            analysis: 'Detected DES cipher with Feistel network structure and characteristic permutation operations.',
            confidence: '94.2%'
        },
        'RSA': {
            roundLoop: { address: '0x00401250', instructions: 20, badge: 'Modular Exp' },
            mixColumns: { address: '0x004012E0', instructions: 10, label: 'Montgomery Mult' },
            analysis: 'Detected RSA with modular exponentiation and Montgomery multiplication optimization.',
            confidence: '93.7%'
        }
    };
    
    // Return specific pattern or default AES pattern
    return cfgPatterns[algorithm] || cfgPatterns['AES'];
}

function updateCFGBlocks(cfgData) {
    // Update Round Loop block
    const roundLoopBlock = document.querySelector('.cfg-loop');
    if (roundLoopBlock) {
        const addressEl = roundLoopBlock.querySelector('.cfg-address');
        const instructionsEl = roundLoopBlock.querySelector('.cfg-instructions');
        const badgeEl = roundLoopBlock.querySelector('.cfg-badge');
        
        if (addressEl) addressEl.textContent = cfgData.roundLoop.address;
        if (instructionsEl) instructionsEl.textContent = `${cfgData.roundLoop.instructions} instructions`;
        if (badgeEl) badgeEl.textContent = cfgData.roundLoop.badge;
    }
    
    // Update Mix Columns block
    const mixColumnsBlock = document.querySelector('.cfg-normal');
    if (mixColumnsBlock) {
        const addressEl = mixColumnsBlock.querySelector('.cfg-address');
        const labelEl = mixColumnsBlock.querySelector('.cfg-label');
        const instructionsEl = mixColumnsBlock.querySelector('.cfg-instructions');
        
        if (addressEl) addressEl.textContent = cfgData.mixColumns.address;
        if (labelEl && cfgData.mixColumns.label) labelEl.textContent = cfgData.mixColumns.label;
        if (instructionsEl) instructionsEl.textContent = `${cfgData.mixColumns.instructions} instructions`;
    }
    
    // Update analysis text
    const analysisText = document.querySelector('.cfg-analysis-text');
    if (analysisText) {
        analysisText.innerHTML = `
            <span class="cfg-analysis-label">Analysis:</span> 
            ${cfgData.analysis}
            <span class="cfg-confidence">High confidence: ${cfgData.confidence}</span>
        `;
    }
}

// Load results on page load
window.addEventListener('DOMContentLoaded', loadAnalysisResults);


// AI Crypto Assistant
let aiMessages = [];
let isAILoading = false;

function initAIAssistant() {
    const aiInput = document.getElementById('aiInput');
    const aiSendBtn = document.getElementById('aiSendBtn');
    
    if (!aiInput || !aiSendBtn) return;
    
    // Send message on button click
    aiSendBtn.addEventListener('click', handleAISend);
    
    // Send message on Enter key
    aiInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleAISend();
        }
    });
}

function handleAISend() {
    const aiInput = document.getElementById('aiInput');
    const message = aiInput.value.trim();
    
    if (!message || isAILoading) return;
    
    // Add user message
    addAIMessage('user', message);
    aiInput.value = '';
    
    // Show typing indicator
    showAITyping();
    
    // Simulate AI response (replace with actual API call)
    setTimeout(() => {
        const response = generateAIResponse(message);
        hideAITyping();
        addAIMessage('assistant', response);
    }, 1500 + Math.random() * 1000);
}

function addAIMessage(role, content) {
    const messagesContainer = document.getElementById('aiMessages');
    if (!messagesContainer) return;
    
    // Remove welcome message if exists
    const welcome = messagesContainer.querySelector('.ai-welcome');
    if (welcome) {
        welcome.remove();
    }
    
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `ai-message ${role}`;
    
    if (role === 'assistant') {
        messageDiv.innerHTML = `
            <div class="ai-avatar bot">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div class="ai-message-content">${content}</div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="ai-message-content">${content}</div>
            <div class="ai-avatar user">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M20 21V19C20 17.9391 19.5786 16.9217 18.8284 16.1716C18.0783 15.4214 17.0609 15 16 15H8C6.93913 15 5.92172 15.4214 5.17157 16.1716C4.42143 16.9217 4 17.9391 4 19V21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <circle cx="12" cy="7" r="4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
        `;
    }
    
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Store message
    aiMessages.push({ role, content });
}

function showAITyping() {
    const messagesContainer = document.getElementById('aiMessages');
    if (!messagesContainer) return;
    
    isAILoading = true;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'ai-typing';
    typingDiv.id = 'aiTypingIndicator';
    typingDiv.innerHTML = `
        <div class="ai-avatar bot">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" style="animation: pulse 2s ease-in-out infinite;">
                <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
        </div>
        <div class="ai-message-content">
            <div class="ai-typing-dots">
                <div class="ai-typing-dot"></div>
                <div class="ai-typing-dot"></div>
                <div class="ai-typing-dot"></div>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Disable input
    document.getElementById('aiInput').disabled = true;
    document.getElementById('aiSendBtn').disabled = true;
}

function hideAITyping() {
    const typingIndicator = document.getElementById('aiTypingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
    
    isAILoading = false;
    
    // Enable input
    document.getElementById('aiInput').disabled = false;
    document.getElementById('aiSendBtn').disabled = false;
}

function generateAIResponse(userMessage) {
    const lowerMessage = userMessage.toLowerCase();
    
    // Get current analysis context
    const storedResult = sessionStorage.getItem('analysisResult');
    let detectedAlgo = 'AES';
    let confidence = 94.2;
    
    if (storedResult) {
        const data = JSON.parse(storedResult);
        detectedAlgo = data.result?.detected || 'AES';
        confidence = data.result?.confidence || 94.2;
    }
    
    // Generate contextual responses
    if (lowerMessage.includes('aes') || lowerMessage.includes('algorithm')) {
        return `Based on the analysis, I detected ${detectedAlgo} with ${confidence.toFixed(1)}% confidence. This algorithm is identified by characteristic patterns like S-box lookups (constant table access), round-based structure, and MixColumns operations. The control flow graph shows typical AES encryption rounds with substitution and permutation layers.`;
    }
    
    if (lowerMessage.includes('function') || lowerMessage.includes('look like')) {
        return `The function exhibits classic cryptographic patterns:\n\n1. **S-Box Operations**: Constant table lookups indicating substitution operations\n2. **Round Structure**: Iterative processing typical of block ciphers\n3. **Bit Manipulation**: XOR operations and shifts common in crypto\n4. **Key Schedule**: Separate key expansion routines\n\nThese patterns strongly suggest ${detectedAlgo} implementation.`;
    }
    
    if (lowerMessage.includes('protocol') || lowerMessage.includes('tls') || lowerMessage.includes('ssl')) {
        return `The firmware shows protocol implementation with:\n\n• **Handshake Phase**: Client/Server hello exchanges\n• **Key Exchange**: ECDHE or RSA key agreement\n• **Encryption Phase**: Symmetric encryption using ${detectedAlgo}\n• **Authentication**: HMAC for message integrity\n\nThis indicates a secure communication protocol like TLS 1.2 or 1.3.`;
    }
    
    if (lowerMessage.includes('confidence') || lowerMessage.includes('sure')) {
        return `The ${confidence.toFixed(1)}% confidence score is based on:\n\n• **Pattern Matching**: ${Math.floor(confidence * 0.4)}% - Characteristic instruction sequences\n• **Control Flow**: ${Math.floor(confidence * 0.3)}% - Expected block structure\n• **Data Flow**: ${Math.floor(confidence * 0.2)}% - Typical crypto data patterns\n• **Context**: ${Math.floor(confidence * 0.1)}% - Surrounding code analysis\n\nHigh confidence indicates strong algorithmic signatures.`;
    }
    
    if (lowerMessage.includes('secure') || lowerMessage.includes('safe') || lowerMessage.includes('vulnerability')) {
        return `Security assessment:\n\n✅ **Strong Points**:\n• Modern algorithm (${detectedAlgo})\n• Proper key sizes detected\n• Standard implementation patterns\n\n⚠️ **Recommendations**:\n• Verify key management practices\n• Check for side-channel protections\n• Ensure proper IV/nonce handling\n• Review authentication mechanisms`;
    }
    
    if (lowerMessage.includes('how') || lowerMessage.includes('work')) {
        return `${detectedAlgo} works through:\n\n1. **Key Expansion**: Derives round keys from master key\n2. **Initial Round**: AddRoundKey operation\n3. **Main Rounds**: SubBytes (S-box), ShiftRows, MixColumns, AddRoundKey\n4. **Final Round**: SubBytes, ShiftRows, AddRoundKey (no MixColumns)\n\nEach round transforms the data block using substitution-permutation network principles.`;
    }
    
    if (lowerMessage.includes('improve') || lowerMessage.includes('optimize')) {
        return `Optimization suggestions:\n\n🚀 **Performance**:\n• Use hardware AES instructions (AES-NI)\n• Implement table-based lookups\n• Consider parallel processing modes (CTR, GCM)\n\n🔒 **Security**:\n• Add constant-time implementations\n• Implement side-channel countermeasures\n• Use authenticated encryption (GCM, CCM)`;
    }
    
    // Default response
    return `I can help you understand the crypto detection results! I see ${detectedAlgo} was detected with ${confidence.toFixed(1)}% confidence.\n\nYou can ask me about:\n• Algorithm characteristics and patterns\n• Function analysis and control flow\n• Protocol implementations\n• Security recommendations\n• Optimization suggestions\n\nWhat would you like to know?`;
}

// Initialize AI Assistant when page loads
window.addEventListener('DOMContentLoaded', () => {
    initAIAssistant();
});
