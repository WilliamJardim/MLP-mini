export default function debugIfSomeNaN(context, varToCheck, callback) {
    let nanValues = [];
    let hasNaN = false;
    varToCheck.forEach((val, valIndex) => {
        if (isNaN(val) || !isFinite(val)) {
            nanValues.push(valIndex);
            console.log('NaN', valIndex);
            hasNaN = true;
        }
    });
    let result = { hasNaN: hasNaN, values: nanValues };
    if (hasNaN) {
        callback.bind(context)(result);
    }
    return result;
}
