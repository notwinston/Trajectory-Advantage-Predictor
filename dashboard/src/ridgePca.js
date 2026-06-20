const EPS = 1e-12;

const finiteNumber = (value) => {
  if (value === null || value === undefined) return 0;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
};

const maybeNumber = (value) => {
  if (value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
};

const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);

const norm = (vector) => Math.sqrt(dot(vector, vector));

const normalize = (vector, fallbackIndex = 0) => {
  const length = norm(vector);
  if (length <= EPS) {
    return vector.map((_, index) => (index === fallbackIndex ? 1 : 0));
  }
  return vector.map((value) => value / length);
};

const matVec = (matrix, vector) => matrix.map((row) => dot(row, vector));

function covariance(centeredRows, width) {
  const matrix = Array.from({ length: width }, () => Array(width).fill(0));
  const denom = Math.max(centeredRows.length - 1, 1);
  for (const row of centeredRows) {
    for (let i = 0; i < width; i += 1) {
      for (let j = i; j < width; j += 1) {
        matrix[i][j] += (row[i] * row[j]) / denom;
      }
    }
  }
  for (let i = 0; i < width; i += 1) {
    for (let j = 0; j < i; j += 1) {
      matrix[i][j] = matrix[j][i];
    }
  }
  return matrix;
}

function initialVector(length, seed) {
  return normalize(
    Array.from({ length }, (_, index) => {
      const n = index + 1;
      return Math.sin(n * (seed + 1) * 1.61803398875) + 0.35 * Math.cos(n * (seed + 3));
    }),
    seed % Math.max(length, 1),
  );
}

function leadingEigenPair(matrix, seed) {
  const width = matrix.length;
  if (!width) return { value: 0, vector: [] };

  let vector = initialVector(width, seed);
  for (let iteration = 0; iteration < 100; iteration += 1) {
    const next = matVec(matrix, vector);
    const nextNorm = norm(next);
    if (nextNorm <= EPS) {
      return { value: 0, vector: normalize(Array(width).fill(0), seed % width) };
    }
    vector = next.map((value) => value / nextNorm);
  }

  const value = Math.max(0, dot(vector, matVec(matrix, vector)));
  const largestIndex = vector.reduce(
    (best, value, index) => (Math.abs(value) > Math.abs(vector[best]) ? index : best),
    0,
  );
  if (vector[largestIndex] < 0) {
    vector = vector.map((value) => -value);
  }
  return { value, vector };
}

function deflate(matrix, eigenValue, eigenVector) {
  return matrix.map((row, i) =>
    row.map((value, j) => value - eigenValue * eigenVector[i] * eigenVector[j]),
  );
}

export function computeRidgePca(run) {
  const snapshots = run?.snapshots || [];
  const featureCount = run?.features?.length || 0;
  const vectors = snapshots.map((snapshot) =>
    Array.from({ length: featureCount }, (_, index) => finiteNumber(snapshot.weights?.[index])),
  );

  if (!vectors.length || !featureCount) {
    return {
      mean: [],
      components: [],
      eigenValues: [0, 0, 0],
      explained: [0, 0, 0],
      projections: [],
      featureLoadings: [[], [], []],
    };
  }

  const mean = Array.from({ length: featureCount }, (_, index) =>
    vectors.reduce((sum, row) => sum + row[index], 0) / vectors.length,
  );
  const centered = vectors.map((row) => row.map((value, index) => value - mean[index]));
  let residual = covariance(centered, featureCount);
  const totalVariance = Math.max(
    residual.reduce((sum, row, index) => sum + Math.max(0, row[index]), 0),
    EPS,
  );

  const components = [];
  const eigenValues = [];
  for (let pc = 0; pc < 3; pc += 1) {
    const pair = leadingEigenPair(residual, pc);
    components.push(pair.vector);
    eigenValues.push(pair.value);
    residual = deflate(residual, pair.value, pair.vector);
  }

  const projections = centered.map((row) =>
    components.map((component) => (component.length ? dot(row, component) : 0)),
  );

  const featureLoadings = components.map((component) =>
    run.features
      .map((feature, index) => ({ feature, loading: component[index] || 0 }))
      .sort((a, b) => Math.abs(b.loading) - Math.abs(a.loading))
      .slice(0, 5),
  );

  return {
    mean,
    components,
    eigenValues,
    explained: eigenValues.map((value) => value / totalVariance),
    projections,
    featureLoadings,
  };
}

export function ridgeStepDetails(run, step) {
  const features = run?.features || [];
  const snapshots = run?.snapshots || [];
  const index = Math.max(0, Math.min(step || 1, snapshots.length) - 1);
  const current = snapshots[index];
  const previous = snapshots[Math.max(0, index - 1)];
  const weights = current?.weights || [];
  const previousWeights = previous?.weights || [];

  const rows = features.map((feature, featureIndex) => {
    const weight = finiteNumber(weights[featureIndex]);
    const prior = index > 0 ? finiteNumber(previousWeights[featureIndex]) : 0;
    return {
      feature,
      weight,
      delta: weight - prior,
    };
  });

  return {
    weightNorm: norm(rows.map((row) => row.weight)),
    topWeights: [...rows].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, 7),
    changedFeatures: rows.sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 7),
  };
}

function orderedRows(run) {
  return (run?.candidates || [])
    .flatMap((candidate) =>
      (candidate.rows || []).map((row) => ({
        ...row,
        candidateId: candidate.id,
      })),
    )
    .sort((a, b) => Number(a.index) - Number(b.index));
}

function rawFeatureValue(raw, feature) {
  const summary = raw?.reward_summary || {};
  if (Object.prototype.hasOwnProperty.call(summary, feature)) {
    return maybeNumber(summary[feature]);
  }
  return maybeNumber(raw?.[feature]);
}

function vectorizeRawRow(run, row, snapshot) {
  return (run?.features || []).map((feature, index) => {
    const value = rawFeatureValue(row.raw, feature);
    const imputed = value === null ? finiteNumber(snapshot?.imputeMeans?.[index]) : value;
    const mu = finiteNumber(snapshot?.mu?.[index]);
    const sd = Math.abs(finiteNumber(snapshot?.sd?.[index])) > EPS ? finiteNumber(snapshot.sd[index]) : 1;
    return (imputed - mu) / sd;
  });
}

function buildTrainingPrefix(run, step, snapshot) {
  return orderedRows(run)
    .slice(0, Math.max(1, Math.min(step || 1, run?.rowCount || 1)))
    .map((row) => ({
      candidateId: row.candidateId,
      x: vectorizeRawRow(run, row, snapshot),
      y: maybeNumber(row.raw?.lift_nll) ?? maybeNumber(row.liftNll) ?? 0,
    }));
}

function weightsFromProjection(mean, components, coords) {
  const out = [...mean];
  for (let axis = 0; axis < 3; axis += 1) {
    const component = components[axis] || [];
    const scale = coords[axis] || 0;
    for (let index = 0; index < out.length; index += 1) {
      out[index] += scale * (component[index] || 0);
    }
  }
  return out;
}

function projectionBounds(projections) {
  const points = projections?.length ? projections : [[0, 0, 0]];
  const mins = [0, 1, 2].map((axis) => Math.min(...points.map((point) => point?.[axis] || 0)));
  const maxs = [0, 1, 2].map((axis) => Math.max(...points.map((point) => point?.[axis] || 0)));
  const spans = maxs.map((max, index) => max - mins[index]);
  const fallbackSpan = Math.max(...spans, 1e-4);
  const pad = fallbackSpan * 0.28;
  const paddedMins = mins.map((value) => value - pad);
  const paddedMaxs = maxs.map((value) => value + pad);
  const center = paddedMins.map((min, index) => (min + paddedMaxs[index]) / 2);
  const radius = Math.max(
    ...paddedMaxs.map((max, index) => Math.abs(max - center[index])),
    ...paddedMins.map((min, index) => Math.abs(min - center[index])),
    1e-4,
  );
  return { mins: paddedMins, maxs: paddedMaxs, center, radius };
}

function gradientForWeights(prefix, weights, base, ridgeLambda = 1.0) {
  const grad = Array(weights.length).fill(0);
  let loss = 0;
  for (const row of prefix) {
    const residual = dot(row.x, weights) + base - row.y;
    loss += residual * residual;
    for (let index = 0; index < weights.length; index += 1) {
      grad[index] += row.x[index] * residual;
    }
  }
  for (let index = 0; index < weights.length; index += 1) {
    loss += ridgeLambda * weights[index] * weights[index];
    grad[index] = 2 * (grad[index] + ridgeLambda * weights[index]);
  }
  return { grad, loss };
}

function projectToComponents(components, vector) {
  return components.map((component) => (component?.length ? dot(component, vector) : 0));
}

export function computeRidgeGradientField(run, pca, step) {
  const snapshots = run?.snapshots || [];
  const index = Math.max(0, Math.min(step || 1, snapshots.length) - 1);
  const snapshot = snapshots[index];
  const weights = (snapshot?.weights || []).map(finiteNumber);
  const components = pca?.components || [];
  const mean = (pca?.mean || []).map(finiteNumber);
  const prefix = buildTrainingPrefix(run, index + 1, snapshot);
  const projections = pca?.projections || [];
  const activeProjection = projections[index] || [0, 0, 0];
  const bounds = projectionBounds(projections);

  const current = gradientForWeights(prefix, weights, finiteNumber(snapshot?.base));
  const currentGradientPc = projectToComponents(components, current.grad);
  const currentGradientNorm = norm(current.grad);
  const currentGradientPcNorm = norm(currentGradientPc);

  const samples = [0, 0.25, 0.5, 0.75, 1];
  const zSamples = [0.15, 0.5, 0.85];
  const vectors = [];
  for (const x of samples) {
    for (const y of samples) {
      for (const z of zSamples) {
        const coords = [x, y, z].map(
          (value, axis) => bounds.mins[axis] + value * (bounds.maxs[axis] - bounds.mins[axis]),
        );
        const probeWeights = weightsFromProjection(mean, components, coords);
        const probe = gradientForWeights(prefix, probeWeights, finiteNumber(snapshot?.base));
        const gradientPc = projectToComponents(components, probe.grad);
        const descent = gradientPc.map((value) => -value);
        vectors.push({
          coords,
          descent,
          gradientNorm: norm(gradientPc),
          loss: probe.loss,
        });
      }
    }
  }

  const maxGradient = Math.max(...vectors.map((vector) => vector.gradientNorm), currentGradientPcNorm, EPS);
  const gradientRows = (run?.features || []).map((feature, featureIndex) => ({
    feature,
    gradient: current.grad[featureIndex] || 0,
  }));

  return {
    index,
    bounds,
    activeProjection,
    trajectory: projections,
    currentLoss: current.loss,
    currentGradientNorm,
    currentGradientPcNorm,
    maxGradient,
    vectors,
    topGradientFeatures: gradientRows.sort((a, b) => Math.abs(b.gradient) - Math.abs(a.gradient)).slice(0, 7),
  };
}
