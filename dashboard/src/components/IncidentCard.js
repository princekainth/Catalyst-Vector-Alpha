import React from 'react';
import RiskBadge from './RiskBadge';

const IncidentCard = ({ incident }) => {
  const {
    id,
    incident_type,
    severity,
    namespace,
    workload,
    pod,
    status,
    recommended_tool,
    risk,
    trace_id,
    classification,
    updated_at
  } = incident;

  const getStatusColor = (status) => {
    switch (status) {
      case 'OPEN': return 'text-red-400 bg-red-400/10 border-red-400/20';
      case 'GATED': return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20';
      case 'RESOLVED': return 'text-green-400 bg-green-400/10 border-green-400/20';
      case 'FAILED': return 'text-gray-400 bg-gray-400/10 border-gray-400/20';
      default: return 'text-blue-400 bg-blue-400/10 border-blue-400/20';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity?.toUpperCase()) {
      case 'CRITICAL': return 'text-red-500 font-bold';
      case 'HIGH': return 'text-orange-400';
      default: return 'text-blue-400';
    }
  };

  return (
    <div className="bg-[#1e293b]/50 border border-slate-700 rounded-xl p-5 mb-4 hover:border-slate-500 transition-all shadow-lg backdrop-blur-sm relative overflow-hidden group">
      {/* Status Glow */}
      <div className={`absolute top-0 right-0 w-1 h-full ${status === 'GATED' ? 'bg-yellow-500' : status === 'OPEN' ? 'bg-red-500' : 'bg-green-500'}`} />

      <div className="flex justify-between items-start mb-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getStatusColor(status)}`}>
              {status}
            </span>
            <span className={`text-sm font-medium ${getSeverityColor(severity)}`}>
              {incident_type}
            </span>
          </div>
          <h4 className="text-slate-200 font-semibold text-lg">{workload || 'Unknown Workload'}</h4>
        </div>
        <div className="text-right">
          <RiskBadge risk={risk} />
          <div className="text-[10px] text-slate-500 mt-1">
            {new Date(updated_at).toLocaleString()}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 mb-4 text-xs">
        <div className="space-y-1">
          <div className="flex justify-between border-b border-slate-700/50 pb-1">
            <span className="text-slate-500">Namespace</span>
            <span className="text-slate-300 font-mono">{namespace || 'default'}</span>
          </div>
          <div className="flex justify-between border-b border-slate-700/50 pb-1">
            <span className="text-slate-500">Pod</span>
            <span className="text-slate-300 font-mono truncate max-w-[120px]">{pod || 'N/A'}</span>
          </div>
        </div>
        <div className="space-y-1">
          <div className="flex justify-between border-b border-slate-700/50 pb-1">
            <span className="text-slate-500">ID</span>
            <span className="text-slate-400 font-mono">{id}</span>
          </div>
          <div className="flex justify-between border-b border-slate-700/50 pb-1">
            <span className="text-slate-500">Trace</span>
            <span className="text-blue-400 font-mono truncate max-w-[120px]">{trace_id || 'N/A'}</span>
          </div>
        </div>
      </div>

      {classification && (
        <div className="bg-slate-900/50 rounded-lg p-3 mb-4 border border-slate-800/50">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1 font-bold">Evidence Preview</div>
          <p className="text-slate-400 text-xs italic leading-relaxed">
            "{classification.length > 120 ? classification.substring(0, 120) + '...' : classification}"
          </p>
        </div>
      )}

      <div className="flex items-center justify-between mt-4 pt-3 border-t border-slate-700/30">
        <div className="flex items-center gap-2">
          {recommended_tool && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-indigo-500/10 border border-indigo-500/30 rounded-md">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
              <span className="text-[11px] text-indigo-300 font-medium">{recommended_tool}</span>
            </div>
          )}
        </div>

        <div className="flex gap-2">
          {trace_id && (
            <button className="px-3 py-1 text-[11px] bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-600 rounded transition-colors font-medium">
              View Trace
            </button>
          )}
          {status === 'GATED' && (
            <div className="flex items-center gap-2 px-3 py-1 bg-yellow-500/20 border border-yellow-500/30 rounded text-yellow-300 text-[11px] font-bold animate-pulse">
              AWAITING APPROVAL
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default IncidentCard;
