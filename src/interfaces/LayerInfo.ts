import UnitInfo from "./UnitInfo";

export default interface LayerInfo{
    title?: string,
    description?: string,
    layerIndex?: number,
    unitsInfo?: UnitInfo[]
}