// MCP Forex Trading Bot - Web Interface

// Global variables
let currentPairs = [];
let mt5AccountInfo = null;
let mt5ActiveTrades = {};

// DOM Ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the dashboard
    fetchMarketStatus();
    fetchGlobalMarkets();
    fetchSignals();
    fetchPairs();
    fetchTradingStatus();
    fetchTradeHistory();
    fetchPerformanceMetrics();
    
    // Set up event listeners
    setupEventListeners();
    
    // Refresh data periodically
    setInterval(fetchMarketStatus, 60000); // Every minute
    setInterval(fetchSignals, 60000); // Every minute
    setInterval(fetchTradingStatus, 30000); // Every 30 seconds
    setInterval(fetchTradeHistory, 60000); // Every minute
    setInterval(fetchPerformanceMetrics, 60000); // Every minute
    
    // Set up tab change listeners
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function (event) {
            if (event.target.id === 'trade-history-tab') {
                fetchTradeHistory();
            } else if (event.target.id === 'performance-tab') {
                fetchPerformanceMetrics();
            }
        });
    });
});

// Fetch and display trading status
function fetchTradingStatus() {
    fetch('/api/trading-status')
        .then(response => response.json())
        .then(data => {
            updateTradingStatus(data);
        })
        .catch(error => {
            console.error('Error fetching trading status:', error);
        });
}

// Fetch market status
function fetchMarketStatus() {
    fetch('/api/market-status')
        .then(response => response.json())
        .then(data => {
            updateMarketStatus(data);
        })
        .catch(error => {
            console.error('Error fetching market status:', error);
        });
}

// Update market status display
function updateMarketStatus(data) {
    const marketStatusElement = document.getElementById('market-status');
    if (marketStatusElement) {
        let html = '';
        if (data.is_open) {
            html = `
                <div class="alert alert-success">
                    <i class="fas fa-check-circle me-2"></i>
                    <strong>Forex Market is OPEN</strong>
                </div>
                <p>${data.market_hours_text || ''}</p>
            `;
        } else {
            html = `
                <div class="alert alert-warning">
                    <i class="fas fa-clock me-2"></i>
                    <strong>Forex Market is CLOSED</strong>
                </div>
                <p>${data.market_hours_text || ''}</p>
                <p>Next open: ${data.next_open || 'Unknown'}</p>
            `;
        }
        marketStatusElement.innerHTML = html;
    }
}

// Fetch global markets
function fetchGlobalMarkets() {
    fetch('/api/global-markets')
        .then(response => response.json())
        .then(data => {
            updateGlobalMarkets(data);
        })
        .catch(error => {
            console.error('Error fetching global markets:', error);
        });
}

// Update global markets display
function updateGlobalMarkets(data) {
    const globalMarketsContainer = document.getElementById('global-markets-container');
    if (globalMarketsContainer && data.centers) {
        let html = '<div class="row">';
        
        // Check if we have any centers data
        if (Object.keys(data.centers).length === 0) {
            globalMarketsContainer.innerHTML = '<div class="alert alert-warning">No global markets data available</div>';
            return;
        }
        
        // Process each trading center
        for (const [centerName, centerData] of Object.entries(data.centers)) {
            const statusClass = centerData.is_open ? 'bg-success text-white' : 'bg-secondary text-white';
            const statusText = centerData.is_open ? 'OPEN' : 'CLOSED';
            const localTime = centerData.local_time || 'Unknown';
            
            html += `
                <div class="col-md-6 mb-3">
                    <div class="card market-center-card ${centerData.is_open ? 'market-open' : 'market-closed'}">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h5 class="card-title mb-0">${centerName}</h5>
                                <span class="badge ${statusClass}">${statusText}</span>
                            </div>
                            <p class="card-text">
                                <strong>Local Time:</strong> ${localTime}
                            </p>
                        </div>
                    </div>
                </div>
            `;
        }
        html += '</div>';
        globalMarketsContainer.innerHTML = html;
    }
}

// Fetch trading signals
function fetchSignals() {
    fetch('/api/signals')
        .then(response => response.json())
        .then(data => {
            updateSignals(data);
        })
        .catch(error => {
            console.error('Error fetching trading signals:', error);
        });
}

// Update signals display
function updateSignals(data) {
    const signalsElement = document.getElementById('trading-signals');
    if (signalsElement && data.signals) {
        if (Object.keys(data.signals).length === 0) {
            signalsElement.innerHTML = '<div class="alert alert-info">No trading signals available</div>';
            return;
        }
        
        let html = '<div class="row">';
        for (const [pair, signal] of Object.entries(data.signals)) {
            const actionClass = signal.action === 'BUY' ? 'success' : (signal.action === 'SELL' ? 'danger' : 'secondary');
            const confidencePercent = (signal.confidence * 100).toFixed(1);
            
            html += `
                <div class="col-md-4 mb-3">
                    <div class="card border-${actionClass}">
                        <div class="card-header bg-${actionClass} text-white">
                            <h5 class="mb-0">${pair}</h5>
                        </div>
                        <div class="card-body">
                            <h4 class="text-center text-${actionClass}">${signal.action}</h4>
                            <div class="progress mb-3">
                                <div class="progress-bar bg-${actionClass}" role="progressbar" style="width: ${confidencePercent}%" aria-valuenow="${confidencePercent}" aria-valuemin="0" aria-valuemax="100">${confidencePercent}%</div>
                            </div>
                            <p><strong>Reason:</strong> ${signal.reason || 'N/A'}</p>
                            <p><small class="text-muted">Updated: ${new Date(signal.timestamp).toLocaleString()}</small></p>
                        </div>
                    </div>
                </div>
            `;
        }
        html += '</div>';
        signalsElement.innerHTML = html;
    }
}

// Fetch currency pairs
function fetchPairs() {
    fetch('/api/pairs')
        .then(response => response.json())
        .then(data => {
            updatePairs(data);
        })
        .catch(error => {
            console.error('Error fetching currency pairs:', error);
        });
}

// Update pairs display
function updatePairs(data) {
    const pairsContainer = document.getElementById('pairs-container');
    if (pairsContainer && data.pairs) {
        if (data.pairs.length === 0) {
            pairsContainer.innerHTML = '<div class="alert alert-info">No currency pairs added yet</div>';
            return;
        }
        
        let html = '';
        for (const pair of data.pairs) {
            html += `
                <div class="badge bg-primary m-1 p-2">
                    ${pair}
                    <button class="btn-close btn-close-white ms-2" data-pair="${pair}" aria-label="Remove"></button>
                </div>
            `;
        }
        pairsContainer.innerHTML = html;
        
        // Add event listeners to remove buttons
        document.querySelectorAll('[data-pair]').forEach(btn => {
            btn.addEventListener('click', function() {
                const pair = this.getAttribute('data-pair');
                removePair(pair);
            });
        });
    }
}

// Update trading status display
function updateTradingStatus(data) {
    // Update account info
    const accountInfoElement = document.getElementById('mt5-account-info');
    if (accountInfoElement && data.account_info) {
        const info = data.account_info;
        let html = `
            <div class="row">
                <div class="col-md-6">
                    <p><strong>Server:</strong> ${info.server || 'N/A'}</p>
                    <p><strong>Balance:</strong> $${info.balance ? info.balance.toFixed(2) : '0.00'}</p>
                    <p><strong>Equity:</strong> $${info.equity ? info.equity.toFixed(2) : '0.00'}</p>
                </div>
                <div class="col-md-6">
                    <p><strong>Leverage:</strong> ${info.leverage || 'N/A'}</p>
                    <p><strong>Free Margin:</strong> $${info.margin_free ? info.margin_free.toFixed(2) : '0.00'}</p>
                    <p><strong>Profit:</strong> <span class="${info.profit >= 0 ? 'text-success' : 'text-danger'}">$${info.profit ? info.profit.toFixed(2) : '0.00'}</span></p>
                </div>
            </div>
        `;
        accountInfoElement.innerHTML = html;
    }
    
    // Update active trades
    const tradesTableBody = document.getElementById('mt5-trades-table-body');
    if (tradesTableBody && data.active_trades) {
        if (Object.keys(data.active_trades).length === 0) {
            tradesTableBody.innerHTML = '<tr><td colspan="8" class="text-center">No active trades</td></tr>';
        } else {
            let html = '';
            for (const [pair, trade] of Object.entries(data.active_trades)) {
                const tradeType = trade.type === 'BUY' ? 
                    '<span class="badge bg-success">BUY</span>' : 
                    '<span class="badge bg-danger">SELL</span>';
                    
                html += `
                    <tr>
                        <td>${pair}</td>
                        <td>${tradeType}</td>
                        <td>${trade.lot_size}</td>
                        <td>${trade.open_price}</td>
                        <td>${new Date(trade.open_time).toLocaleString()}</td>
                        <td>${trade.stop_loss || 'N/A'}</td>
                        <td>${trade.take_profit || 'N/A'}</td>
                        <td>
                            <button class="btn btn-sm btn-danger close-trade-btn" data-ticket="${trade.ticket}">
                                <i class="fas fa-times"></i> Close
                            </button>
                        </td>
                    </tr>
                `;
            }
            tradesTableBody.innerHTML = html;
            
            // Add event listeners to close buttons
            document.querySelectorAll('.close-trade-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const ticket = this.getAttribute('data-ticket');
                    closeTrade(ticket);
                });
            });
        }
    }
    
    // Update trade pair select
    const tradePairSelect = document.getElementById('trade-pair-select');
    if (tradePairSelect && data.available_pairs) {
        // Save current selection
        const currentSelection = tradePairSelect.value;
        
        // Clear options except the first one
        while (tradePairSelect.options.length > 1) {
            tradePairSelect.remove(1);
        }
        
        // Add pairs
        data.available_pairs.forEach(pair => {
            const option = document.createElement('option');
            option.value = pair;
            option.textContent = pair;
            tradePairSelect.appendChild(option);
        });
        
        // Restore selection if possible
        if (currentSelection && data.available_pairs.includes(currentSelection)) {
            tradePairSelect.value = currentSelection;
        }
    }
}

// Fetch trade history from MongoDB
function fetchTradeHistory() {
    fetch('/api/trade-history')
        .then(response => response.json())
        .then(data => {
            updateTradeHistory(data);
        })
        .catch(error => {
            console.error('Error fetching trade history:', error);
        });
}

// Update trade history display
function updateTradeHistory(data) {
    const tradeHistoryBody = document.getElementById('trade-history-body');
    if (tradeHistoryBody) {
        if (!data.trades || data.trades.length === 0) {
            tradeHistoryBody.innerHTML = '<tr><td colspan="10" class="text-center">No trade history available</td></tr>';
        } else {
            let html = '';
            data.trades.forEach(trade => {
                const tradeType = trade.type === 'BUY' ? 
                    '<span class="badge bg-success">BUY</span>' : 
                    '<span class="badge bg-danger">SELL</span>';
                    
                const status = trade.status === 'open' ? 
                    '<span class="badge bg-primary">OPEN</span>' : 
                    '<span class="badge bg-secondary">CLOSED</span>';
                    
                const profit = trade.profit ? 
                    `<span class="${parseFloat(trade.profit) >= 0 ? 'text-success' : 'text-danger'}">$${parseFloat(trade.profit).toFixed(2)}</span>` : 
                    'N/A';
                    
                html += `
                    <tr>
                        <td>${trade.ticket}</td>
                        <td>${trade.pair}</td>
                        <td>${tradeType}</td>
                        <td>${trade.lot_size}</td>
                        <td>${trade.open_price}</td>
                        <td>${trade.close_price || 'N/A'}</td>
                        <td>${new Date(trade.open_time).toLocaleString()}</td>
                        <td>${trade.close_time ? new Date(trade.close_time).toLocaleString() : 'N/A'}</td>
                        <td>${profit}</td>
                        <td>${status}</td>
                    </tr>
                `;
            });
            tradeHistoryBody.innerHTML = html;
        }
    }
}

// Fetch performance metrics from MongoDB
function fetchPerformanceMetrics() {
    fetch('/api/performance')
        .then(response => response.json())
        .then(data => {
            updatePerformanceMetrics(data);
        })
        .catch(error => {
            console.error('Error fetching performance metrics:', error);
        });
}

// Update performance metrics display
function updatePerformanceMetrics(data) {
    if (!data.performance) {
        return;
    }
    
    const performance = data.performance;
    
    // Update summary metrics
    document.getElementById('total-trades').textContent = performance.total_trades || '0';
    document.getElementById('win-rate').textContent = performance.win_rate ? `${(performance.win_rate * 100).toFixed(1)}%` : '0%';
    document.getElementById('total-profit').textContent = performance.total_profit ? `$${performance.total_profit.toFixed(2)}` : '$0.00';
    document.getElementById('profit-factor').textContent = performance.profit_factor ? performance.profit_factor.toFixed(2) : '0.00';
    
    // Update win/loss ratio progress bar
    const winningTrades = performance.winning_trades || 0;
    const losingTrades = performance.losing_trades || 0;
    const totalTrades = winningTrades + losingTrades;
    
    if (totalTrades > 0) {
        const winPercent = (winningTrades / totalTrades) * 100;
        const lossPercent = (losingTrades / totalTrades) * 100;
        
        const winProgress = document.getElementById('win-progress');
        const lossProgress = document.getElementById('loss-progress');
        
        winProgress.style.width = `${winPercent}%`;
        winProgress.textContent = `${winPercent.toFixed(1)}%`;
        winProgress.setAttribute('aria-valuenow', winPercent);
        
        lossProgress.style.width = `${lossPercent}%`;
        lossProgress.textContent = `${lossPercent.toFixed(1)}%`;
        lossProgress.setAttribute('aria-valuenow', lossPercent);
        
        document.getElementById('winning-trades').textContent = winningTrades;
        document.getElementById('losing-trades').textContent = losingTrades;
    }
    
    // Update average trade metrics
    document.getElementById('average-profit').textContent = performance.average_win ? `$${performance.average_win.toFixed(2)}` : '$0.00';
    document.getElementById('average-loss').textContent = performance.average_loss ? `$${performance.average_loss.toFixed(2)}` : '$0.00';
}

// Set up event listeners
function setupEventListeners() {
    // Add pair button
    const addPairBtn = document.getElementById('addPairBtn');
    if (addPairBtn) {
        addPairBtn.addEventListener('click', function() {
            const pairInput = document.getElementById('pairInput');
            const pair = pairInput.value.trim().toUpperCase();
            if (pair) {
                addPair(pair);
                pairInput.value = '';
            }
        });
    }
    
    // Execute trade button
    const executeTradeBtn = document.getElementById('execute-trade-btn');
    if (executeTradeBtn) {
        executeTradeBtn.addEventListener('click', function() {
            const pairSelect = document.getElementById('trade-pair-select');
            const actionSelect = document.getElementById('trade-action-select');
            const lotSizeInput = document.getElementById('trade-lot-size');
            
        });
    }
    
    // Close all trades button
    const closeAllTradesBtn = document.getElementById('close-all-trades-btn');
    if (closeAllTradesBtn) {
        closeAllTradesBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to close all active trades?')) {
                closeAllTrades();
            }
        });
    }
}

// Load market status
function loadMarketStatus() {
    fetch('/api/market-status')
        .then(response => response.json())
        .then(data => {
            updateMarketStatus(data);
        })
        .catch(error => {
            console.error('Error loading market status:', error);
        });
}

// Update market status in the UI
function updateMarketStatus(data) {
    const statusIndicator = document.getElementById('market-status-indicator');
    const statusText = document.getElementById('market-status-text');
    const hoursText = document.getElementById('market-hours-text');
    
    if (data.is_open) {
        statusIndicator.className = 'status-indicator status-open';
        statusText.textContent = 'Market Open';
    } else {
        statusIndicator.className = 'status-indicator status-closed';
        statusText.textContent = 'Market Closed';
    }
    
    hoursText.textContent = data.market_hours_text;
}

// Load global markets
function loadGlobalMarkets() {
    fetch('/api/global-markets')
        .then(response => response.json())
        .then(data => {
            updateGlobalMarkets(data);
        })
        .catch(error => {
            console.error('Error loading global markets:', error);
        });
}

// Update global markets in the UI
function updateGlobalMarkets(data) {
    const container = document.getElementById('global-markets-container');
    
    if (data.centers_status) {
        let html = '<div class="row">';
        
        for (const center in data.centers_status) {
            const status = data.centers_status[center];
            const isOpen = status.is_open;
            const statusClass = isOpen ? 'market-open' : 'market-closed';
            const statusText = isOpen ? 'Open' : 'Closed';
            const statusIcon = isOpen ? 'fa-door-open text-success' : 'fa-door-closed text-danger';
            
            html += `
                <div class="col-md-6 col-lg-3 mb-3">
                    <div class="market-center-card ${statusClass}">
                        <h5>${center}</h5>
                        <div class="d-flex align-items-center">
                            <i class="fas ${statusIcon} me-2"></i>
                            <span>${statusText}</span>
                        </div>
                        <div class="mt-2">
                            <small>Local Time: ${status.local_time}</small>
                        </div>
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
    } else {
        container.innerHTML = '<div class="alert alert-warning">No global markets data available</div>';
    }
}

// This section is intentionally left empty as these functions are now defined above

// Update trading signals in the UI
function updateTradingSignals() {
    const container = document.getElementById('signals-container');
    
    if (Object.keys(currentSignals).length > 0) {
        // Sort signals by confidence (highest first)
        const sortedSignals = Object.entries(currentSignals)
            .filter(([_, signal]) => signal.action && ['BUY', 'SELL', 'WEAK BUY', 'WEAK SELL', 'HOLD'].includes(signal.action))
            .sort((a, b) => (b[1].confidence || 0) - (a[1].confidence || 0));
        
        if (sortedSignals.length > 0) {
            let html = '<div class="row">';
            
            for (const [pair, signal] of sortedSignals) {
                const action = signal.action || 'HOLD';
                const confidence = (signal.confidence || 0) * 100;
                const price = signal.price || 'N/A';
                const timestamp = signal.timestamp || 'N/A';
                
                // Determine signal class and icon
                let signalClass, actionText, actionIcon;
                
                switch (action) {
                    case 'BUY':
                        signalClass = 'signal-buy';
                        actionText = '<span class="text-success">BUY</span>';
                        actionIcon = 'fa-arrow-up text-success';
                        break;
                    case 'SELL':
                        signalClass = 'signal-sell';
                        actionText = '<span class="text-danger">SELL</span>';
                        actionIcon = 'fa-arrow-down text-danger';
                        break;
                    case 'WEAK BUY':
                        signalClass = 'signal-weak-buy';
                        actionText = '<span class="text-warning">WEAK BUY</span>';
                        actionIcon = 'fa-arrow-up text-warning';
                        break;
                    case 'WEAK SELL':
                        signalClass = 'signal-weak-sell';
                        actionText = '<span class="text-warning">WEAK SELL</span>';
                        actionIcon = 'fa-arrow-down text-warning';
                        break;
                    default:
                        signalClass = 'signal-hold';
                        actionText = '<span class="text-secondary">HOLD</span>';
                        actionIcon = 'fa-minus text-secondary';
                }
                
                html += `
                    <div class="col-md-6 col-lg-4 mb-3">
                        <div class="card signal-card ${signalClass}">
                            <div class="card-body">
                                <div class="d-flex justify-content-between align-items-center mb-2">
                                    <h5 class="card-title mb-0">${pair}</h5>
                                    <div>
                                        <i class="fas ${actionIcon} me-1"></i>
                                        ${actionText}
                                    </div>
                                </div>
                                <div class="mb-2">
                                    <strong>Price:</strong> ${price}
                                </div>
                                <div class="mb-2">
                                    <strong>Confidence:</strong> ${confidence.toFixed(1)}%
                                </div>
                                <div class="mb-2">
                                    <strong>Time:</strong> ${timestamp}
                                </div>
                                
                                <!-- Technical Indicators -->
                                ${renderTechnicalIndicators(signal)}
                                
                                <!-- ICT Model Analysis -->
                                ${renderICTAnalysis(signal)}
                            </div>
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            container.innerHTML = html;
        } else {
            container.innerHTML = '<div class="alert alert-info">No actionable signals at the moment. All pairs are in HOLD status.</div>';
        }
    } else {
        container.innerHTML = '<div class="alert alert-warning">No trading signals available yet. Please wait for the next analysis cycle.</div>';
    }
}

// Render ICT model analysis
function renderICTAnalysis(signal) {
    if (!signal.indicators || !signal.indicators.ict) {
        return '';
    }
    
    const ict = signal.indicators.ict;
    let html = '<div class="mt-3"><strong>ICT Model Analysis:</strong></div>';
    
    // Market structure
    if (ict.market_structure) {
        const structure = ict.market_structure.structure || 'Unknown';
        let structureClass = 'text-secondary';
        
        if (structure === 'bullish') structureClass = 'text-success';
        else if (structure === 'bearish') structureClass = 'text-danger';
        
        html += `
            <div class="mt-2">
                <small>Market Structure: <span class="${structureClass}">${structure.toUpperCase()}</span></small>
            </div>
        `;
    }
    
    // Key levels
    if (ict.key_levels) {
        // Order blocks
        const orderBlocks = ict.key_levels.order_blocks || [];
        if (orderBlocks.length > 0) {
            html += `<div class="mt-2"><small>Order Blocks: ${orderBlocks.length} identified</small></div>`;
        }
        
        // Fair value gaps
        const fvgs = ict.key_levels.fair_value_gaps || [];
        if (fvgs.length > 0) {
            html += `<div class="mt-2"><small>Fair Value Gaps: ${fvgs.length} identified</small></div>`;
        }
        
        // Breaker blocks
        const breakerBlocks = ict.key_levels.breaker_blocks || [];
        if (breakerBlocks.length > 0) {
            html += `<div class="mt-2"><small>Breaker Blocks: ${breakerBlocks.length} identified</small></div>`;
        }
        
        // Optimal trade entry
        const ote = ict.key_levels.optimal_trade_entry || [];
        if (ote.length > 0) {
            html += `<div class="mt-2"><small>OTE Zones: ${ote.length} active</small></div>`;
        }
    }
    
    // ICT Confidence
    if (ict.confidence !== undefined) {
        const confidence = ict.confidence * 100;
        let confidenceClass = 'text-secondary';
        
        if (confidence > 70) confidenceClass = 'text-success';
        else if (confidence > 40) confidenceClass = 'text-warning';
        else if (confidence > 0) confidenceClass = 'text-danger';
        
        html += `
            <div class="mt-2">
                <small>ICT Confidence: <span class="${confidenceClass}">${confidence.toFixed(1)}%</span></small>
            </div>
        `;
    }
    
    return html;
}

// Render technical indicators
function renderTechnicalIndicators(signal) {
    if (!signal.indicators) {
        return '';
    }
    
    let html = '<div class="mt-3"><strong>Technical Indicators:</strong></div><div class="row">';
    
    // RSI
    if (signal.indicators.rsi !== undefined) {
        const rsi = signal.indicators.rsi;
        let rsiClass = 'text-secondary';
        
        if (rsi < 30) rsiClass = 'text-success';
        else if (rsi > 70) rsiClass = 'text-danger';
        
        html += `
            <div class="col-6">
                <small>RSI: <span class="${rsiClass}">${rsi.toFixed(2)}</span></small>
            </div>
        `;
    }
    
    // MACD
    if (signal.indicators.macd) {
        const macd = signal.indicators.macd;
        html += `
            <div class="col-6">
                <small>MACD: ${macd.macd?.toFixed(4) || 'N/A'}</small>
            </div>
        `;
    }
    
    // Stochastic
    if (signal.indicators.stochastic) {
        const stoch = signal.indicators.stochastic;
        html += `
            <div class="col-6">
                <small>Stoch %K: ${stoch.k?.toFixed(2) || 'N/A'}</small>
            </div>
            <div class="col-6">
                <small>%D: ${stoch.d?.toFixed(2) || 'N/A'}</small>
            </div>
        `;
    }
    
    html += '</div>';
    return html;
}

// Load currency pairs
function loadCurrencyPairs() {
    fetch('/api/pairs')
        .then(response => response.json())
        .then(data => {
            currentPairs = data.pairs || [];
            updateCurrencyPairs();
        })
        .catch(error => {
            console.error('Error loading currency pairs:', error);
        });
}

// Update currency pairs in the UI
function updateCurrencyPairs() {
    const container = document.getElementById('pairs-container');
    
    if (currentPairs.length > 0) {
        let html = '<div class="d-flex flex-wrap">';
        
        for (const pair of currentPairs) {
            html += `
                <div class="pair-badge">
                    ${pair}
                    <span class="remove-pair" onclick="removeCurrencyPair('${pair}')">
                        <i class="fas fa-times-circle"></i>
                    </span>
                </div>
            `;
        }
        
        html += '</div>';
        container.innerHTML = html;
    } else {
        container.innerHTML = '<div class="alert alert-warning">No currency pairs are being monitored.</div>';
    }
}

// Add currency pair
function addCurrencyPair(pair) {
    // Validate the pair format (should be 6 characters, e.g., EURUSD)
    if (pair.length !== 6) {
        showToast('Invalid currency pair format. Please use format like EURUSD.', 'danger');
        return;
    }
    
    // Check if pair already exists
    if (currentPairs.includes(pair)) {
        showToast(`Currency pair ${pair} is already being monitored.`, 'warning');
        return;
    }
    
    fetch('/api/add-pair', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ pair: pair })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(data.message, 'success');
                loadCurrencyPairs();
            } else {
                showToast(data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error adding currency pair:', error);
            showToast('Error adding currency pair', 'danger');
        });
}

// Remove currency pair
function removeCurrencyPair(pair) {
    fetch('/api/remove-pair', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ pair: pair })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(data.message, 'success');
                loadCurrencyPairs();
            } else {
                showToast(data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error removing currency pair:', error);
            showToast('Error removing currency pair', 'danger');
        });
}

// This section is intentionally left empty as these functions are now defined above

// Update MT5 trading status in the UI
function updateMT5TradingStatus() {
    const statusContainer = document.getElementById('mt5-status-container');
    const accountInfoSection = document.getElementById('mt5-account-info');
    const activeTradesSection = document.getElementById('mt5-active-trades');
    const manualTradeSection = document.getElementById('mt5-manual-trade');
    
    if (mt5Status.mt5_enabled) {
        // MT5 is enabled
        statusContainer.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle me-2"></i>
                MT5 Trading is enabled. 
                <span class="badge bg-${mt5Status.auto_trading ? 'success' : 'warning'} ms-2">
                    Auto Trading: ${mt5Status.auto_trading ? 'Enabled' : 'Disabled'}
                </span>
            </div>
        `;
        
        // Show account info section
        accountInfoSection.style.display = 'block';
        activeTradesSection.style.display = 'block';
        manualTradeSection.style.display = 'block';
        
        // Update account info
        if (mt5Status.account_info) {
            mt5AccountInfo = mt5Status.account_info;
            
            document.getElementById('mt5-account-name').textContent = mt5AccountInfo.name;
            document.getElementById('mt5-account-login').textContent = mt5AccountInfo.login;
            document.getElementById('mt5-account-server').textContent = mt5AccountInfo.server;
            document.getElementById('mt5-account-leverage').textContent = `1:${mt5AccountInfo.leverage}`;
            
            document.getElementById('mt5-account-balance').textContent = `${mt5AccountInfo.balance} ${mt5AccountInfo.currency}`;
            document.getElementById('mt5-account-equity').textContent = `${mt5AccountInfo.equity} ${mt5AccountInfo.currency}`;
            document.getElementById('mt5-account-margin').textContent = `${mt5AccountInfo.margin} ${mt5AccountInfo.currency}`;
            document.getElementById('mt5-account-free-margin').textContent = `${mt5AccountInfo.free_margin} ${mt5AccountInfo.currency}`;
        }
        
        // Update active trades
        mt5ActiveTrades = mt5Status.active_trades || {};
        updateActiveTrades();
        
        // Update pair select dropdown
        updatePairSelectDropdown();
    } else {
        // MT5 is disabled
        statusContainer.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                MT5 Trading is not enabled. Please check your configuration.
            </div>
        `;
        
        // Hide sections
        accountInfoSection.style.display = 'none';
        activeTradesSection.style.display = 'none';
        manualTradeSection.style.display = 'none';
    }
}

// Update active trades table
function updateActiveTrades() {
    const tableBody = document.getElementById('mt5-trades-table-body');
    
    if (Object.keys(mt5ActiveTrades).length > 0) {
        let html = '';
        
        for (const pair in mt5ActiveTrades) {
            const trade = mt5ActiveTrades[pair];
            const tradeType = trade.type;
            const typeClass = tradeType === 'BUY' ? 'text-success' : 'text-danger';
            
            html += `
                <tr>
                    <td>${pair}</td>
                    <td class="${typeClass}">${tradeType}</td>
                    <td>${trade.lot_size}</td>
                    <td>${trade.open_price}</td>
                    <td>${new Date(trade.open_time).toLocaleString()}</td>
                    <td>${trade.stop_loss || '-'}</td>
                    <td>${trade.take_profit || '-'}</td>
                    <td>
                        <button class="btn btn-sm btn-danger" onclick="closeTrade('${pair}')">
                            <i class="fas fa-times"></i> Close
                        </button>
                    </td>
                </tr>
            `;
        }
        
        tableBody.innerHTML = html;
    } else {
        tableBody.innerHTML = '<tr><td colspan="8" class="text-center">No active trades</td></tr>';
    }
}

// Update pair select dropdown
function updatePairSelectDropdown() {
    const pairSelect = document.getElementById('trade-pair-select');
    
    if (pairSelect) {
        // Clear existing options except the first one
        while (pairSelect.options.length > 1) {
            pairSelect.remove(1);
        }
        
        // Add current pairs
        for (const pair of currentPairs) {
            const option = document.createElement('option');
            option.value = pair;
            option.textContent = pair;
            pairSelect.appendChild(option);
        }
    }
}

// Execute a trade
function executeTrade(pair, action, lotSize) {
    fetch('/api/execute-trade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            pair: pair,
            action: action,
            lot_size: lotSize
        })
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(data.message, 'success');
                // Reload trading status to show the new trade
                loadMT5TradingStatus();
            } else {
                showToast(data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error executing trade:', error);
            showToast('Error executing trade', 'danger');
        });
}

// Close a specific trade
function closeTrade(pair) {
    if (confirm(`Are you sure you want to close the ${pair} trade?`)) {
        // For now, we'll use the close all trades endpoint
        // In a future update, we could add a specific close trade endpoint
        closeAllTrades();
    }
}

// Close all trades
function closeAllTrades() {
    fetch('/api/close-all-trades', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showToast(data.message, 'success');
                // Reload trading status to update the trades list
                loadMT5TradingStatus();
            } else {
                showToast(data.message, 'danger');
            }
        })
        .catch(error => {
            console.error('Error closing trades:', error);
            showToast('Error closing trades', 'danger');
        });
}

// Show toast notification
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast show bg-${type} text-white`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    // Toast content
    toast.innerHTML = `
        <div class="toast-header bg-${type} text-white">
            <strong class="me-auto">MCP Forex Bot</strong>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
        <div class="toast-body">
            ${message}
        </div>
    `;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
    
    // Close button functionality
    const closeButton = toast.querySelector('.btn-close');
    if (closeButton) {
        closeButton.addEventListener('click', () => {
            toast.remove();
        });
    }
}
