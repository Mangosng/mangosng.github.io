// Stock API module - calls Supabase Edge Function for predictions
import { SUPABASE_URL, SUPABASE_ANON_KEY, supabase } from './supabase';

/**
 * Format ticker for the correct market
 */
export function formatTicker(ticker, market) {
  const cleanTicker = ticker.toUpperCase().trim();
  if (market === 'TSX') {
    return cleanTicker.endsWith('.TO') ? cleanTicker : `${cleanTicker}.TO`;
  }
  return cleanTicker;
}

/**
 * Call the Supabase Edge Function to get stock prediction
 */
export async function predictStock(ticker, market, targetDate) {
  const response = await fetch(`${SUPABASE_URL}/functions/v1/predict-stock`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${SUPABASE_ANON_KEY}`,
    },
    body: JSON.stringify({
      ticker,
      market,
      target_date: targetDate,
    }),
  });
  
  const data = await response.json();
  
  if (!response.ok) {
    throw new Error(data.error || 'Failed to get prediction');
  }
  
  return data;
}

/**
 * Get company info (simple fetch from quote endpoint)
 */
export async function fetchCompanyInfo(ticker, market) {
  const formattedTicker = formatTicker(ticker, market);
  
  // For now, just return the ticker as name
  // This could be enhanced with a separate Edge Function if needed
  return {
    name: formattedTicker,
    ticker: formattedTicker,
  };
}

/**
 * Calculate days between two dates
 */
export function daysBetween(date1, date2) {
  const d1 = new Date(date1);
  const d2 = new Date(date2);
  const diffTime = Math.abs(d2 - d1);
  return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
}

/**
 * Get historical accuracy for a ticker or global accuracy if not tracked
 */
export async function getHistoricalAccuracy(ticker) {
  if (!ticker) return null;
  const cleanTicker = ticker.toUpperCase().replace('.TO', ''); 

  // 1. Try to get specific stats for this ticker
  const { count: specificTotal, error: countErr } = await supabase
    .from('predictions')
    .select('is_correct', { count: 'exact', head: true })
    .eq('ticker', cleanTicker)
    .not('is_correct', 'is', null);

  if (countErr) {
    console.error('Error fetching specific stats:', countErr);
    return null;
  }

  if (specificTotal > 0) {
     // Get correct count
     const { count: specificCorrect } = await supabase
      .from('predictions')
      .select('is_correct', { count: 'exact', head: true })
      .eq('ticker', cleanTicker)
      .eq('is_correct', true);
      
     return {
       ticker: cleanTicker,
       accuracy: (specificCorrect / specificTotal) * 100,
       correct: specificCorrect,
       total: specificTotal,
       isGlobal: false
     };
  }

  // 2. If no data for ticker, get Global Stats
  const { count: globalTotal } = await supabase
    .from('predictions')
    .select('is_correct', { count: 'exact', head: true })
    .not('is_correct', 'is', null);

  if (!globalTotal || globalTotal === 0) return null;

  const { count: globalCorrect } = await supabase
    .from('predictions')
    .select('is_correct', { count: 'exact', head: true })
    .eq('is_correct', true);

  return {
    ticker: 'S&P 500 (Global)',
    accuracy: (globalCorrect / globalTotal) * 100,
    correct: globalCorrect,
    total: globalTotal,
    isGlobal: true
  };
}
