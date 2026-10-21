export interface CarModel {
  model: string
  brand: string
  body_type: string
  fuels: string[]
}

export interface NumericSpec {
  min: number
  max: number
  default: number
  step: number
}

export interface Metrics {
  mae: number
  r2: number
  mape: number
  n_test: number
}

export interface Schema {
  priceUnit: string
  refYear: number
  outputName: string
  inputOrder: string[]
  numeric: { year: NumericSpec; km_driven: NumericSpec }
  categorical: { fuel: string[]; transmission: string[]; owner: string[] }
  models: CarModel[]
  metrics: Metrics
}

export interface CarInput {
  brand: string
  model: string
  body_type: string
  fuel: string
  transmission: string
  owner: string
  year: number
  km_driven: number
}
