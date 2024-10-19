var jaFoi = {};
export default function notifyIfhasNaN(title, varToCheck, callback) {
    let nanValues = [];
    let hasNaN = false;
    varToCheck.forEach((val, valIndex) => {
        if (val instanceof Array) {
            let resultSub = notifyIfhasNaN(title + '_array', val);
            nanValues = [...resultSub.values, nanValues];
            hasNaN = resultSub.hasNaN;
        }
        else {
            if (isNaN(val)) {
                nanValues.push(valIndex);
                if (!jaFoi[title]) {
                    console.warn(title, 'NaN', valIndex, 'please insert debugger');
                    jaFoi[title] = true;
                }
                hasNaN = true;
            }
        }
    });
    let result = { hasNaN: hasNaN, values: nanValues };
    if (hasNaN && callback) {
        callback(result);
    }
    return result;
}
