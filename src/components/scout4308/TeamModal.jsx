import { useState } from 'react';
import { upsertScoutingData, SHOOTER_TYPES, CLIMB_OPTIONS } from '../../lib/scouting';

const TeamModal = ({ team, teamStats, teamYearStats, scoutingData, eventKey, onClose, onSave }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    shooter_type: scoutingData?.shooter_type || '',
    hopper_capacity: scoutingData?.hopper_capacity || '',
    autos: scoutingData?.autos || '',
    feeding: scoutingData?.feeding || false,
    climb: scoutingData?.climb || '',
    trench: scoutingData?.trench || false,
  });
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const dataToSave = {
        team_number: team.team_number,
        competition_key: eventKey,
        shooter_type: formData.shooter_type || null,
        hopper_capacity: formData.hopper_capacity ? parseInt(formData.hopper_capacity) : null,
        autos: formData.autos || null,
        feeding: formData.feeding,
        climb: formData.climb || null,
        trench: formData.trench,
        updated_at: new Date().toISOString(),
      };
      const saved = await upsertScoutingData(dataToSave);
      onSave(saved);
      setIsEditing(false);
    } catch (err) {
      setError('Failed to save data');
      console.error(err);
    } finally {
      setSaving(false);
    }
  };

  // Get EPA display info
  const getEpaDisplay = () => {
    // Check for 2026 event EPA
    if (teamStats?.epa?.total) {
      return { epa: teamStats.epa.total, year: 2026, isCurrent: true };
    }
    // Check for 2026 year stats
    if (teamYearStats?.dataYear === 2026 && teamYearStats?.epa?.total_points?.mean) {
      return { 
        epa: teamYearStats.epa.total_points.mean, 
        year: 2026, 
        isCurrent: true,
        normalized: teamYearStats.epa?.norm,
      };
    }
    return null;
  };

  // Get ranking and normalized EPA from most recent year
  const getHistoricalStats = () => {
    if (!teamYearStats) return null;
    
    const rank = teamYearStats.epa?.ranks?.total?.rank;
    const normalized = teamYearStats.epa?.norm;
    const year = teamYearStats.dataYear;
    
    return { rank, normalized, year };
  };

  const epaDisplay = getEpaDisplay();
  const historicalStats = getHistoricalStats();

  return (
    <div className="fixed inset-0 bg-ink/80 flex items-center justify-center z-50 p-4">
      <div className="bg-canvas border border-structure w-full max-w-lg max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="border-b border-structure p-4 flex justify-between items-start">
          <div>
            <h2 className="text-xl font-bold uppercase tracking-terminal">
              {team.team_number}
            </h2>
            <p className="text-sm text-ink/70 uppercase tracking-terminal">
              {team.nickname}
            </p>
            <p className="text-xs tracking-terminal text-ink/50 mt-1">
              {[team.city, team.state_prov, team.country].filter(Boolean).join(', ')}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-ink hover:bg-highlight hover:text-invert px-3 py-1 border border-structure transition-none"
          >
            [ X ]
          </button>
        </div>

        {/* Stats Section */}
        <div className="border-b border-structure p-4 space-y-2">
          <p className="text-xs uppercase tracking-terminal text-ink/70 mb-2">STATBOTICS DATA</p>
          
          <div className="flex flex-wrap gap-2">
            {/* 2026 EPA (only if available) */}
            {epaDisplay && (
              <span className="text-xs uppercase tracking-terminal px-2 py-1 border border-structure">
                EPA: {epaDisplay.epa.toFixed(1)}
              </span>
            )}
            
            {/* Most recent year global rank */}
            {historicalStats?.rank && (
              <span className="text-xs uppercase tracking-terminal px-2 py-1 border border-structure">
                {historicalStats.year} RANK: #{historicalStats.rank}
              </span>
            )}
            
            {/* Most recent normalized EPA */}
            {historicalStats?.normalized && (
              <span className="text-xs uppercase tracking-terminal px-2 py-1 border border-structure">
                {historicalStats.year} NORM: {historicalStats.normalized.toFixed(0)}
              </span>
            )}
          </div>
          
          {!epaDisplay && !historicalStats?.rank && !historicalStats?.normalized && (
            <p className="text-xs text-ink/50 uppercase tracking-terminal">
              [ NO STATBOTICS DATA AVAILABLE ]
            </p>
          )}
        </div>

        {/* Scouting Data Section */}
        <div className="p-4 space-y-4">
          {!isEditing ? (
            <>
              {/* View Mode */}
              <div className="space-y-3">
                <p className="text-xs uppercase tracking-terminal text-ink/70 mb-2">SCOUTING DATA</p>
                
                {scoutingData ? (
                  <div className="space-y-2 text-sm uppercase tracking-terminal">
                    {scoutingData.shooter_type && (
                      <p>SHOOTER: {scoutingData.shooter_type}</p>
                    )}
                    {scoutingData.hopper_capacity && (
                      <p>HOPPER: {scoutingData.hopper_capacity}</p>
                    )}
                    {scoutingData.climb && (
                      <p>CLIMB: {scoutingData.climb}</p>
                    )}
                    <p>FEEDING: {scoutingData.feeding ? 'YES' : 'NO'}</p>
                    <p>TRENCH: {scoutingData.trench ? 'YES' : 'NO'}</p>
                    {scoutingData.autos && (
                      <div>
                        <p className="text-ink/70">AUTOS:</p>
                        <p className="text-xs whitespace-pre-wrap normal-case">{scoutingData.autos}</p>
                      </div>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-ink/50 uppercase tracking-terminal">
                    [ NOT YET SCOUTED ]
                  </p>
                )}
              </div>
              
              {/* Edit Button */}
              <button
                onClick={() => setIsEditing(true)}
                className="w-full bg-highlight text-invert p-4 text-sm uppercase tracking-terminal font-semibold hover:bg-ink transition-none"
              >
                [ EDIT SCOUTING DATA ]
              </button>
            </>
          ) : (
            <>
              {/* Edit Mode */}
              {/* Shooter Type */}
              <div>
                <label className="block text-xs uppercase tracking-terminal mb-2 text-ink/70">
                  SHOOTER TYPE
                </label>
                <select
                  value={formData.shooter_type}
                  onChange={(e) => handleChange('shooter_type', e.target.value)}
                  className="w-full bg-canvas border border-structure p-3 text-sm font-mono"
                >
                  <option value="">-- Select --</option>
                  {SHOOTER_TYPES.map((type) => (
                    <option key={type} value={type}>{type}</option>
                  ))}
                </select>
              </div>

              {/* Hopper Capacity */}
              <div>
                <label className="block text-xs uppercase tracking-terminal mb-2 text-ink/70">
                  HOPPER CAPACITY
                </label>
                <input
                  type="number"
                  value={formData.hopper_capacity}
                  onChange={(e) => handleChange('hopper_capacity', e.target.value)}
                  className="w-full bg-canvas border border-structure p-3 text-sm font-mono"
                  placeholder="Number of game pieces"
                  min="0"
                />
              </div>

              {/* Climb */}
              <div>
                <label className="block text-xs uppercase tracking-terminal mb-2 text-ink/70">
                  CLIMB LEVEL
                </label>
                <div className="flex gap-2 flex-wrap">
                  {CLIMB_OPTIONS.map((option) => (
                    <button
                      key={option}
                      onClick={() => handleChange('climb', option)}
                      className={`flex-1 min-w-[60px] p-3 text-sm uppercase tracking-terminal border transition-none ${
                        formData.climb === option
                          ? 'bg-highlight text-invert border-highlight'
                          : 'border-structure hover:bg-highlight hover:text-invert'
                      }`}
                    >
                      {formData.climb === option ? '[*]' : '[ ]'} {option}
                    </button>
                  ))}
                </div>
              </div>

              {/* Feeding */}
              <div
                onClick={() => handleChange('feeding', !formData.feeding)}
                className={`p-3 border cursor-pointer transition-none ${
                  formData.feeding
                    ? 'bg-highlight text-invert border-highlight'
                    : 'border-structure hover:bg-highlight hover:text-invert'
                }`}
              >
                <span className="text-sm uppercase tracking-terminal font-semibold">
                  {formData.feeding ? '[*]' : '[ ]'} FEEDING CAPABILITY
                </span>
              </div>

              {/* Trench */}
              <div
                onClick={() => handleChange('trench', !formData.trench)}
                className={`p-3 border cursor-pointer transition-none ${
                  formData.trench
                    ? 'bg-highlight text-invert border-highlight'
                    : 'border-structure hover:bg-highlight hover:text-invert'
                }`}
              >
                <span className="text-sm uppercase tracking-terminal font-semibold">
                  {formData.trench ? '[*]' : '[ ]'} CAN GO UNDER TRENCH
                </span>
              </div>

              {/* Autos */}
              <div>
                <label className="block text-xs uppercase tracking-terminal mb-2 text-ink/70">
                  AUTONOMOUS PATHS
                </label>
                <textarea
                  value={formData.autos}
                  onChange={(e) => handleChange('autos', e.target.value)}
                  className="w-full bg-canvas border border-structure p-3 text-sm font-mono min-h-[100px] resize-none"
                  placeholder="Describe auto paths..."
                />
              </div>

              {/* Error */}
              {error && (
                <p className="text-sm text-red-600 uppercase tracking-terminal">
                  [ {error} ]
                </p>
              )}

              {/* Action Buttons */}
              <div className="flex gap-2">
                <button
                  onClick={() => setIsEditing(false)}
                  className="flex-1 border border-structure p-4 text-sm uppercase tracking-terminal font-semibold hover:bg-highlight hover:text-invert transition-none"
                >
                  [ CANCEL ]
                </button>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="flex-1 bg-highlight text-invert p-4 text-sm uppercase tracking-terminal font-semibold hover:bg-ink transition-none disabled:opacity-50"
                >
                  {saving ? '[ SAVING... ]' : '[ SAVE ]'}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default TeamModal;
