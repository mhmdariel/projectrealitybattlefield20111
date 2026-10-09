"use strict";

/*
 * QISC Evidence Sentinel — Integrated Prototype
 * Qur'anic textual references are review markers, not verdicts
 * about a person's identity, guilt, or eternal destiny.
 */

const QISC = Object.freeze({
  namespace: "QISC.EvidenceSentinel",
  version: "1.1.0",

  metricSpace: Object.freeze({
    representation: "sparse-metric",
    dimensionModel: "extensible",
    allocation: "finite-observed-coordinates"
  }),

  triggerName: "EVIDENCE_REVIEW_REQUIRED"
});

const PRINCIPLES = Object.freeze({
  obstructing_worship: {
    references: ["Qur'an 2:114"],
    description: "Review alleged obstruction of worship."
  },

  obstructing_hajj: {
    references: ["Qur'an 22:25"],
    description: "Review alleged obstruction involving the Sacred Mosque."
  },

  coercive_spousal_separation: {
    references: ["Qur'an 2:231", "Qur'an 4:19"],
    description: "Review alleged unjust treatment in a marriage."
  },

  unjust_masjid_exclusion: {
    references: ["Qur'an 2:114"],
    description: "Review alleged unjust exclusion from a mosque."
  }
});

// Qur'an 2:102 textual marker.
const QURAN_MARKERS = Object.freeze({
  harutMarut: Object.freeze({
    reference: "Qur'an 2:102",
    names: Object.freeze(["Harut", "Marut"]),
    markerType: "QURAN_TEXTUAL_REFERENCE",
    purpose: "CONTEXTUAL_EVIDENCE_REVIEW",
    selectsPeopleForTargeting: false,
    triggersPunishment: false
  })
});

class MetricSpace {
  constructor() {
    this.points = new Map();
  }

  setPoint(id, coordinates = {}) {
    const point = new Map();

    for (const [dimension, value] of Object.entries(coordinates)) {
      if (!Number.isFinite(value)) {
        throw new TypeError(`Invalid coordinate: ${dimension}`);
      }

      if (value !== 0) point.set(dimension, value);
    }

    this.points.set(id, point);
  }

  distance(idA, idB) {
    const a = this.points.get(idA);
    const b = this.points.get(idB);

    if (!a || !b) {
      throw new Error("Unknown metric-space point.");
    }

    const dimensions = new Set([...a.keys(), ...b.keys()]);
    let sum = 0;

    for (const key of dimensions) {
      const delta = (a.get(key) ?? 0) - (b.get(key) ?? 0);
      sum += delta * delta;
    }

    return Math.sqrt(sum);
  }
}

function applyHarutMarutMarker(record) {
  const references = Array.isArray(record.quranReferences)
    ? record.quranReferences
    : [];

  const matched = references.includes(
    QURAN_MARKERS.harutMarut.reference
  );

  return {
    caseId: record.id,
    marker: matched ? QURAN_MARKERS.harutMarut : null,
    reviewRequired: matched,
    targetingAuthorized: false
  };
}

class EvidenceSentinel {
  constructor() {
    this.metricSpace = new MetricSpace();
    this.cases = new Map();
    this.lockedCases = new Set();
    this.auditLog = [];
  }

  register(record) {
    if (!record || typeof record.id !== "string" || !record.id.trim()) {
      throw new TypeError("A valid case ID is required.");
    }

    if (this.cases.has(record.id)) {
      throw new Error(`Duplicate case ID: ${record.id}`);
    }

    if (!Object.hasOwn(PRINCIPLES, record.category)) {
      throw new Error("Unknown evidence category.");
    }

    if (!Array.isArray(record.sources) || record.sources.length === 0) {
      throw new Error("Evidence sources are required.");
    }

    const safeRecord = {
      ...record,
      sources: [...record.sources],
      quranReferences: [...(record.quranReferences ?? [])]
    };

    this.cases.set(record.id, safeRecord);

    this.auditLog.push({
      event: "CASE_REGISTERED",
      caseId: record.id,
      timestamp: new Date().toISOString()
    });

    return { ...safeRecord };
  }

  scan(zone) {
    if (!Array.isArray(zone)) {
      throw new TypeError("The scan zone must be an array.");
    }

    return zone.map(record => {
      const knownCase = this.cases.get(record.id);

      if (!knownCase) {
        return {
          caseId: record.id ?? null,
          status: "UNREGISTERED",
          reviewRequired: true
        };
      }

      const principle = PRINCIPLES[knownCase.category];
      const marker = applyHarutMarutMarker(knownCase);

      return {
        caseId: knownCase.id,
        category: knownCase.category,
        principle: principle.description,
        principleReferences: principle.references,
        textualMarker: marker,
        status: knownCase.status ?? "alleged",
        result: "HUMAN_REVIEW_REQUIRED",
        verdictOnEternalDestiny: null
      };
    });
  }

  lockOn(caseId, reviewerId) {
    if (!this.cases.has(caseId)) {
      throw new Error("Unknown case; registration is required first.");
    }

    if (typeof reviewerId !== "string" || !reviewerId.trim()) {
      throw new Error("An accountable reviewer is required.");
    }

    this.lockedCases.add(caseId);

    const event = {
      event: "CASE_LOCKED_FOR_REVIEW",
      caseId,
      reviewerId,
      trigger: QISC.triggerName,
      automaticTargeting: false,
      automaticPunishment: false,
      timestamp: new Date().toISOString()
    };

    this.auditLog.push(event);
    return { ...event };
  }

  getAuditLog() {
    return this.auditLog.map(entry => ({ ...entry }));
  }
}

// Demonstration with synthetic records.
const sentinel = new EvidenceSentinel();

const zone = [
  {
    id: "CASE-001",
    category: "obstructing_hajj",
    status: "alleged",
    sources: ["synthetic-source-A"],
    quranReferences: ["Qur'an 22:25"]
  },
  {
    id: "CASE-002",
    category: "unjust_masjid_exclusion",
    status: "under_review",
    sources: ["synthetic-source-B"],
    quranReferences: ["Qur'an 2:114", "Qur'an 2:102"]
  }
];

for (const record of zone) {
  sentinel.register(record);
}

console.log("Zone scan:");
console.log(JSON.stringify(sentinel.scan(zone), null, 2));

console.log("Review trigger:");
console.log(
  JSON.stringify(
    sentinel.lockOn("CASE-002", "reviewer-001"),
    null,
    2
  )
);

console.log("Audit log:");
console.log(JSON.stringify(sentinel.getAuditLog(), null, 2));
