
var jaFoi = {};

export default function notifyIfhasNaN( title:string, varToCheck:any[], callback?:(resultV:any)=>void ): any{
    let nanValues:number[] = [];
    let hasNaN:boolean = false;

    varToCheck.forEach( (val, valIndex)=>{
        if( val instanceof Array ){ 
            let resultSub = notifyIfhasNaN( title+'_array', val );
            nanValues = [...resultSub.values, nanValues];
            hasNaN    = resultSub.hasNaN;
            
        }else{

            if( isNaN(val) ){
                nanValues.push( valIndex );
                if( !jaFoi[title] ){
                    console.warn( title, 'NaN', valIndex, 'please insert debugger' );
                    jaFoi[title] = true;
                }
                hasNaN = true;
            }
            
        }
    });

    let result:any = { hasNaN: hasNaN, values: nanValues };

    if(hasNaN && callback){
        callback(result);
    }

    return result;
}