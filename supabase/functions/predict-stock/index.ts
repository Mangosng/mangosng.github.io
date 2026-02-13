// Supabase Edge Function: predict-stock
// Trains MLR model (via Python service) and returns stock price prediction

import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface MacroData {
  fedFunds: Map<string, number>;
  cpi: Map<string, number>;
}

interface ProcessedData {
  close: number;
  sma_20: number;
  volatility: number;
  log_return_5d: number;
  log_return_10d: number;
  volume_z_score_60d: number;
  fed_funds: number;
  cpi: number;
}

// Format ticker for market
function formatTicker(ticker: string, market: string): string {
  const clean = ticker.toUpperCase().trim();
  if (market === "TSX") {
    return clean.endsWith(".TO") ? clean : `${clean}.TO`;
  }
  return clean;
}

// Calculate Simple Moving Average
function calculateSMA(data: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
  }
  return result;
}

// Calculate Log Returns (Momentum)
function calculateLogReturn(prices: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < prices.length; i++) {
    if (i < period) {
      result.push(null);
    } else {
      result.push(Math.log(prices[i] / prices[i - period]));
    }
  }
  return result;
}

// Calculate Rolling Z-Score
function calculateRollingZScore(data: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      const window = data.slice(i - period + 1, i + 1);
      const mean = window.reduce((a, b) => a + b, 0) / period;
      const variance = window.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period;
      const std = Math.sqrt(variance) || 1; // Avoid divide by zero
      result.push((data[i] - mean) / std);
    }
  }
  return result;
}

// Calculate Volatility (Std Dev of Log Returns)
function calculateVolatility(prices: number[], period: number = 20): (number | null)[] {
  const result: (number | null)[] = [];
  
  for (let i = 0; i < prices.length; i++) {
    if (i < period) {
      result.push(null);
    } else {
      const returns: number[] = [];
      for (let j = i - period + 1; j <= i; j++) {
        const logReturn = Math.log(prices[j] / prices[j - 1]);
        returns.push(logReturn);
      }
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
      result.push(Math.sqrt(variance));
    }
  }
  return result;
}

// Calculate 95% Confidence Interval bounds
function calculateBounds(currentPrice: number, dailyVol: number, daysAhead: number) {
  const sigma = dailyVol * Math.sqrt(daysAhead);
  return {
    lower: currentPrice * (1 - 2 * sigma),
    upper: currentPrice * (1 + 2 * sigma),
  };
}

// Fetch stock data from Polygon.io
async function fetchStockData(ticker: string, apiKey: string): Promise<StockData[]> {
  // Calculate date range (2 years ago to today)
  const today = new Date();
  const twoYearsAgo = new Date();
  twoYearsAgo.setFullYear(today.getFullYear() - 2);
  
  const fromDate = twoYearsAgo.toISOString().split('T')[0];
  const toDate = today.toISOString().split('T')[0];
  
  const url = `https://api.polygon.io/v2/aggs/ticker/${ticker}/range/1/day/${fromDate}/${toDate}?adjusted=true&sort=asc&limit=50000&apiKey=${apiKey}`;
  const response = await fetch(url);
  const data = await response.json();
  
  if (data.status !== "OK" && data.status !== "ok" && data.status !== "DELAYED") {
    throw new Error(`Polygon error: ${data.error || data.status}`);
  }
  
  if (!data.results || data.results.length === 0) {
    throw new Error("No data returned from Polygon");
  }
  
  // Map Polygon results to our format
  return data.results.map((d: any) => ({
    date: new Date(d.t).toISOString().split('T')[0],
    open: d.o,
    high: d.h,
    low: d.l,
    close: d.c,
    volume: d.v,
  }));
}

// Fetch historical macro data from FRED
async function fetchMacroData(apiKey: string): Promise<MacroData> {
  // Fetch last 2 years of observations
  const today = new Date();
  const twoYearsAgo = new Date();
  twoYearsAgo.setFullYear(today.getFullYear() - 2);
  const startDate = twoYearsAgo.toISOString().split('T')[0];

  // Using two separate fetch calls, handling parallel promise
  const [fedFundsRes, cpiRes] = await Promise.all([
    fetch(`https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key=${apiKey}&file_type=json&sort_order=desc&observation_start=${startDate}`),
    fetch(`https://api.stlouisfed.org/fred/series/observations?series_id=CPIAUCSL&api_key=${apiKey}&file_type=json&sort_order=desc&observation_start=${startDate}`),
  ]);
  
  if (!fedFundsRes.ok) throw new Error(`FRED API Error (Fed Funds): ${fedFundsRes.statusText}`);
  if (!cpiRes.ok) throw new Error(`FRED API Error (CPI): ${cpiRes.statusText}`);

  const fedFundsData = await fedFundsRes.json();
  const cpiData = await cpiRes.json();
  
  // Helper to process FRED observations into a map { date: value }
  const processObservations = (data: any) => {
    const map = new Map<string, number>();
    if (data.observations) {
      data.observations.forEach((obs: any) => {
        map.set(obs.date, parseFloat(obs.value));
      });
    }
    return map;
  };

  return {
    fedFunds: processObservations(fedFundsData),
    cpi: processObservations(cpiData),
  };
}

serve(async (req: Request) => {
  // Handle CORS
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const { ticker, market, target_date } = await req.json();
    
    if (!ticker || !market || !target_date) {
      throw new Error("Missing required fields: ticker, market, target_date");
    }
    
    const formattedTicker = formatTicker(ticker, market);
    const targetDate = new Date(target_date);
    const today = new Date();
    const daysAhead = Math.ceil((targetDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
    
    if (daysAhead <= 0) {
      throw new Error("Target date must be in the future");
    }
    
    // Get API keys from environment
    const polygonKey = Deno.env.get("POLYGON_API_KEY");
    const fredKey = Deno.env.get("FRED_API_KEY");
    
    if (!polygonKey) {
      throw new Error("POLYGON_API_KEY not configured");
    }
    
    // Fetch data
    const [stockData, macroData] = await Promise.all([
      fetchStockData(formattedTicker, polygonKey),
      fetchMacroData(fredKey || ""),
    ]);
    
    if (stockData.length < 200) { // Need more history for 60d Z-score
      throw new Error("Insufficient historical data");
    }
    
    // Calculate indicators
    const closes = stockData.map((d) => d.close);
    const volumes = stockData.map((d) => d.volume);
    
    const sma20 = calculateSMA(closes, 20);
    const volatility = calculateVolatility(closes, 20);
    const logReturn5d = calculateLogReturn(closes, 5);
    const logReturn10d = calculateLogReturn(closes, 10);
    const volumeZScore60d = calculateRollingZScore(volumes, 60);
    
    // Helper to get latest macro value before a date
    const getMacroValue = (map: Map<string, number>, dateStr: string) => {
      // Crude but effective for small datasets (24 months): iterate and find latest
      let latestVal: number | null = null;
      let latestDate = "";
      for (const [d, v] of map.entries()) {
        if (d <= dateStr && d > latestDate) {
          latestDate = d;
          latestVal = v;
        }
      }
      return latestVal;
    };

    // Build processed data
    const processedData: ProcessedData[] = [];
    // Start from 60 to ensure all indicators are available (Volume Z-Score needs 60)
    for (let i = 60; i < stockData.length; i++) {
      if (sma20[i] == null || volatility[i] == null || logReturn5d[i] == null || logReturn10d[i] == null || volumeZScore60d[i] == null) continue;
      
      const dateStr = stockData[i].date;
      const fedFunds = getMacroValue(macroData.fedFunds, dateStr);
      const cpi = getMacroValue(macroData.cpi, dateStr);

      if (fedFunds === null || cpi === null) {
        // If we can't find macro data for a specific date, it might be due to lag.
        // However, if we miss it for *all* datapoints, that's a problem.
        // For now, let's skip rows where we can't align data, rather than falsifying it.
        // Or if it's the *latest* data point, we might fail hard.
        // Given this loop builds training data, skipping a few days is better than bad data.
        continue;
      }

      processedData.push({
        close: closes[i],
        sma_20: sma20[i]!,
        volatility: volatility[i]!,
        log_return_5d: logReturn5d[i]!,
        log_return_10d: logReturn10d[i]!,
        volume_z_score_60d: volumeZScore60d[i]!,
        fed_funds: fedFunds, 
        cpi: cpi,
      });
    }
    
    if (processedData.length < daysAhead + 30) {
      // If we filtered out too much data due to missing macro indicators, this will trip
      throw new Error("Insufficient aligned data (Stock + Macro) for prediction");
    }

    // Prepare payload for Python Service
    const trainingDataPoints = processedData.slice(0, -daysAhead).map(row => ({
      date: "2023-01-01", 
      close: row.close,
      volume: row.volume_z_score_60d, // Mapping Z-score to 'volume' input
      sma_20: row.sma_20,
      volatility: row.volatility,
      fed_funds: row.fed_funds,
      cpi: row.cpi
    }));

    const latest = processedData[processedData.length - 1];
    const currentFeatures = {
      close: latest.close,
      volume: latest.volume_z_score_60d, // Mapping Z-score
      sma_20: latest.sma_20,
      volatility: latest.volatility,
      fed_funds: latest.fed_funds,
      cpi: latest.cpi
    };

    const mlServiceUrl = "https://stock-predictor-ml-1jmf.onrender.com/train-and-predict";
    
    const mlResponse = await fetch(mlServiceUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        training_data: trainingDataPoints,
        current_features: currentFeatures,
        days_ahead: daysAhead
      })
    });

    if (!mlResponse.ok) {
      const errText = await mlResponse.text();
      throw new Error(`ML Service Error: ${errText}`);
    }

    const result = await mlResponse.json();
    
    let prediction = result.predicted_price;
    const rSquared = result.r_squared || 0.0;
    const hitRate = result.hit_rate || 0.0;
    const featureImportance = result.feature_importance;
    
    // Apply dynamic clamping (2-sigma rule)
    const { lower, upper } = calculateBounds(latest.close, latest.volatility, daysAhead);
    prediction = Math.max(lower, Math.min(prediction, upper)); 
    
    // Save to database
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const supabase = createClient(supabaseUrl, supabaseKey);
    
    const { error: dbError } = await supabase.from("stock_predictions").insert({
      ticker: formattedTicker,
      market,
      target_date,
      predicted_price: prediction,
      lower_bound: lower,
      upper_bound: upper,
      input_features: {
        ...latest,
        r_squared: rSquared,
        hit_rate: hitRate,
        training_samples: trainingDataPoints.length,
      },
    });
    
    if (dbError) {
      console.error("Database error:", dbError);
    }
    
    return new Response(
      JSON.stringify({
        ticker: formattedTicker,
        market,
        target_date,
        days_ahead: daysAhead,
        current_price: latest.close,
        predicted_price: Math.round(prediction * 100) / 100,
        lower_bound: Math.round(lower * 100) / 100,
        upper_bound: Math.round(upper * 100) / 100,
        currency: market === "TSX" ? "CAD" : "USD",
        r_squared: Math.round(rSquared * 1000) / 1000,
        hit_rate: Math.round(hitRate * 1000) / 1000,
        training_samples: trainingDataPoints.length,
        volatility: Math.round(latest.volatility * 10000) / 10000,
        feature_importance: featureImportance,
      }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 200,
      }
    );
  } catch (error: any) {
    return new Response(
      JSON.stringify({ error: error.message || "Unknown error" }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 400,
      }
    );
  }
});
