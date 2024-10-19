
export default function debugIfSomeNaN( context:any, varToCheck:number[], callback?:(resultV:any)=>void ): any{
    let nanValues:number[] = [];
    let hasNaN:boolean = false;

    varToCheck.forEach( (val, valIndex)=>{
        if( isNaN(val) || !isFinite(val) ){
            nanValues.push( valIndex );
            console.warn( 'NaN', valIndex, 'please insert debugger' );
            hasNaN = true;
        }
    });

    let result:any = { hasNaN: hasNaN, values: nanValues };

    if(hasNaN && callback){
        callback.bind(context)(result);
    }

    return result;
}