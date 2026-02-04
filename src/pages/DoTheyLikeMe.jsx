import { useState, useMemo } from 'react';
import questions from '../data/questions.json';

const DoTheyLikeMe = () => {
    const [checkedState, setCheckedState] = useState({});

    const handleOnChange = (id) => {
        setCheckedState(prevState => ({
            ...prevState,
            [id]: !prevState[id]
        }));
    };

    const totalScore = useMemo(() => {
        return questions.reduce((total, q) => {
            if (checkedState[q.id]) {
                return total + q.value;
            }
            return total;
        }, 0);
    }, [checkedState]);

    const getResult = (score) => {
        if (score <= -300) return "YOU ARE HARASSING THEM. STOP.";
        if (score <= -150) return "GET A GRIP.";
        if (score <= -50) return "PURE DELUSION IF YOU THINK THEY LIKE YOU.";
        if (score <= 0) return "FRIEND ZONE. DO NOT MAKE A MOVE.";
        if (score <= 75) return "PROBABLY JUST FRIENDS.";
        if (score <= 175) return "THEY LOWKEY LIKE YOU. PROCEED WITH CAUTION.";
        if (score <= 275) return "THEY LIKE YOU BUT ARE SHY.";
        if (score <= 375) return "THEY LIKE YOU. OBVIOUS ABOUT IT.";
        if (score <= 450) return "THEY MIGHT BE IN LOVE WITH YOU.";
        return "GET MARRIED IMMEDIATELY.";
    };

    const result = getResult(totalScore);
    const hasInteracted = Object.values(checkedState).some(val => val);
    const displayResult = hasInteracted ? result : "[ NOT ENOUGH DATA ]";

    return (
        <main className="flex-grow flex flex-col items-center px-6 pb-20">
            {/* TITLE */}
            <h1 className="text-2xl font-bold mt-10 uppercase tracking-terminal text-center text-ink">
                [ DO THEY LIKE ME? ]
            </h1>

            <h2 className="text-center text-ink/60 mt-4 mb-10 leading-relaxed max-w-2xl text-sm uppercase tracking-terminal">
                Check all that apply. If unsure, answer no.
            </h2>

            <form className="w-full max-w-2xl flex flex-col gap-2">
                {questions.map((q) => (
                    <div 
                      className={`flex items-start space-x-3 p-3 border border-structure hover:bg-ink hover:text-invert hover:border-ink transition-none cursor-pointer ${checkedState[q.id] ? 'bg-ink text-invert border-ink' : ''}`} 
                      key={q.id}
                      onClick={() => handleOnChange(q.id)}
                    >
                        <span className="text-sm font-semibold w-8 flex-shrink-0">
                            {checkedState[q.id] ? '[*]' : '[ ]'}
                        </span>
                        <label htmlFor={q.id} className="text-sm cursor-pointer select-none uppercase tracking-terminal">
                            {q.text}
                        </label>
                    </div>
                ))}
            </form>

            <div className="mt-16 text-center sticky bottom-0 bg-canvas border-t border-structure w-full py-6">
                <h3 className="text-2xl font-bold text-ink mb-2 uppercase tracking-terminal">
                  SCORE: {totalScore}
                </h3>
                <h3 className="text-sm text-ink/80 uppercase tracking-terminal">
                  {displayResult}
                </h3>
            </div>

        </main>
    );
};

export default DoTheyLikeMe;
