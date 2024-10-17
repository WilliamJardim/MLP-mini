export default function isDecimalNumber( x:Number ): boolean{
    return String(x).indexOf('.') != -1 ? true : false;
}