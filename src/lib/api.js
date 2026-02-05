import { TBA_API_KEY, TBA_BASE_URL, STATBOTICS_BASE_URL, TEAM_KEY } from './supabase';

// Fetch team's events for a given year
export async function getTeamEvents(year = 2025) {
  const response = await fetch(`${TBA_BASE_URL}/team/${TEAM_KEY}/events/${year}`, {
    headers: {
      'X-TBA-Auth-Key': TBA_API_KEY,
    },
  });
  if (!response.ok) throw new Error('Failed to fetch events');
  return response.json();
}

// Fetch all teams at an event
export async function getEventTeams(eventKey) {
  const response = await fetch(`${TBA_BASE_URL}/event/${eventKey}/teams`, {
    headers: {
      'X-TBA-Auth-Key': TBA_API_KEY,
    },
  });
  if (!response.ok) throw new Error('Failed to fetch teams');
  return response.json();
}

// Fetch matches for an event
export async function getEventMatches(eventKey) {
  const response = await fetch(`${TBA_BASE_URL}/event/${eventKey}/matches/simple`, {
    headers: {
      'X-TBA-Auth-Key': TBA_API_KEY,
    },
  });
  if (!response.ok) throw new Error('Failed to fetch matches');
  const matches = await response.json();
  // Sort by match number
  return matches.sort((a, b) => {
    if (a.comp_level !== b.comp_level) {
      const order = { qm: 1, ef: 2, qf: 3, sf: 4, f: 5 };
      return (order[a.comp_level] || 0) - (order[b.comp_level] || 0);
    }
    return a.match_number - b.match_number;
  });
}

// Fetch event info
export async function getEventInfo(eventKey) {
  const response = await fetch(`${TBA_BASE_URL}/event/${eventKey}`, {
    headers: {
      'X-TBA-Auth-Key': TBA_API_KEY,
    },
  });
  if (!response.ok) throw new Error('Failed to fetch event info');
  return response.json();
}

// Fetch Statbotics data for an event (EPA and predictions)
export async function getStatboticsEvent(eventKey) {
  try {
    const response = await fetch(`${STATBOTICS_BASE_URL}/matches?event=${eventKey}`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    console.error('Statbotics error:', error);
    return null;
  }
}

// Fetch team EPA from Statbotics
export async function getTeamEPA(teamNumber, year = 2025) {
  try {
    const response = await fetch(`${STATBOTICS_BASE_URL}/team_year/${teamNumber}/${year}`);
    if (!response.ok) return null;
    return response.json();
  } catch (error) {
    console.error('Statbotics team error:', error);
    return null;
  }
}

// Fetch all team EPAs for an event
export async function getEventTeamStats(eventKey) {
  try {
    const response = await fetch(`${STATBOTICS_BASE_URL}/team_events?event=${eventKey}`);
    if (!response.ok) return [];
    return response.json();
  } catch (error) {
    console.error('Statbotics event teams error:', error);
    return [];
  }
}

// Fetch team year stats with fallback to previous years
export async function getTeamYearStats(teamNumber) {
  const years = [2026, 2025, 2024, 2023];
  
  for (const year of years) {
    try {
      const response = await fetch(`${STATBOTICS_BASE_URL}/team_year/${teamNumber}/${year}`);
      if (response.ok) {
        const data = await response.json();
        return {
          ...data,
          dataYear: year,
          hasCurrentYearData: year === 2026,
        };
      }
    } catch (error) {
      continue;
    }
  }
  return null;
}

// Fetch stats for multiple teams (batch with fallback)
export async function getTeamsYearStats(teamNumbers) {
  const results = {};
  
  // Try to batch fetch 2026 data first
  try {
    const promises = teamNumbers.map(async (num) => {
      const stats = await getTeamYearStats(num);
      if (stats) {
        results[num] = stats;
      }
    });
    
    // Process in batches to avoid overwhelming the API
    const batchSize = 10;
    for (let i = 0; i < promises.length; i += batchSize) {
      await Promise.all(promises.slice(i, i + batchSize));
    }
  } catch (error) {
    console.error('Error fetching team stats:', error);
  }
  
  return results;
}
