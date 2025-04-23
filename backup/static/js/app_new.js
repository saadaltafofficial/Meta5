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
    const globalMarketsElement = document.getElementById('global-markets');
    if (globalMarketsElement && data.markets) {
        let html = '<div class="row">';
        for (const market of data.markets) {
            const changeClass = market.change >= 0 ? 'text-success' : 'text-danger';
            const changeIcon = market.change >= 0 ? 'fa-arrow-up' : 'fa-arrow-down';
            html += `
                <div class="col-md-4 mb-3">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="card-title">${market.name}</h5>
                            <p class="card-text">
                                <strong>${market.index}:</strong> ${market.value}
                                <span class="${changeClass}">
                                    <i class="fas ${changeIcon}"></i>
                                    ${Math.abs(market.change)}%
                                </span>
                            </p>
                            <p class="card-text"><small class="text-muted">Status: ${market.status}</small></p>
                        </div>
                    </div>
                </div>
            `;
        }
        html += '</div>';
        globalMarketsElement.innerHTML = html;
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
            currentPairs = data.pairs || [];
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

// Fetch and display trading status
function fetchTradingStatus() {
    fetch('/api/trading-status')
        .then(response => response.json())
        .then(data => {
            mt5AccountInfo = data.account_info || null;
            mt5ActiveTrades = data.active_trades || {};
            updateTradingStatus(data);
        })
        .catch(error => {
            console.error('Error fetching trading status:', error);
        });
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
    if (tradePairSelect && currentPairs.length > 0) {
        // Save current selection
        const currentSelection = tradePairSelect.value;
        
        // Clear options except the first one
        while (tradePairSelect.options.length > 1) {
            tradePairSelect.remove(1);
        }
        
        // Add pairs
        currentPairs.forEach(pair => {
            const option = document.createElement('option');
            option.value = pair;
            option.textContent = pair;
            tradePairSelect.appendChild(option);
        });
        
        // Restore selection if possible
        if (currentSelection && currentPairs.includes(currentSelection)) {
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
    document.getElementById('average-profit').textContent = performance.average_profit ? `$${performance.average_profit.toFixed(2)}` : '$0.00';
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
            
            const pair = pairSelect.value;
            const action = actionSelect.value;
            const lotSize = parseFloat(lotSizeInput.value);
            
            if (!pair) {
                alert('Please select a currency pair');
                return;
            }
            
            if (isNaN(lotSize) || lotSize <= 0) {
                alert('Please enter a valid lot size');
                return;
            }
            
            executeTrade(pair, action, lotSize);
        });
    }
    
    // Close all trades button
    const closeAllTradesBtn = document.getElementById('close-all-trades-btn');
    if (closeAllTradesBtn) {
        closeAllTradesBtn.addEventListener('click', function() {
            closeAllTrades();
        });
    }
}

// Add currency pair
function addPair(pair) {
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
            fetchPairs(); // Refresh pairs list
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error adding pair:', error);
        alert('Error adding pair. Please try again.');
    });
}

// Remove currency pair
function removePair(pair) {
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
            fetchPairs(); // Refresh pairs list
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error removing pair:', error);
        alert('Error removing pair. Please try again.');
    });
}

// Execute trade
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
            alert('Trade executed successfully!');
            fetchTradingStatus(); // Refresh trading status
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error executing trade:', error);
        alert('Error executing trade. Please try again.');
    });
}

// Close a specific trade
function closeTrade(ticket) {
    if (confirm('Are you sure you want to close this trade?')) {
        closeAllTrades(); // For now, we'll use the close all trades endpoint
    }
}

// Close all trades
function closeAllTrades() {
    if (!confirm('Are you sure you want to close all active trades?')) {
        return;
    }
    
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
            alert(data.message);
            fetchTradingStatus(); // Refresh trading status
        } else {
            alert('Error: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error closing trades:', error);
        alert('Error closing trades. Please try again.');
    });
}
