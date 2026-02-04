import { useState, useMemo } from 'react';
import compatibilityData from '../data/compatibility.json';

const AreYouCompatible = () => {
    const [checkedState, setCheckedState] = useState({});
    const { sections, results } = compatibilityData;

    const handleOnChange = (id) => {
        setCheckedState(prevState => ({
            ...prevState,
            [id]: !prevState[id]
        }));
    };

    // Calculate score per section
    const sectionScores = useMemo(() => {
        return sections.map(section => {
            const score = section.questions.reduce((total, q) => {
                if (checkedState[q.id]) {
                    return total + q.value;
                }
                return total;
            }, 0);
            return { id: section.id, name: section.name, score };
        });
    }, [checkedState, sections]);

    // Calculate total score
    const totalScore = useMemo(() => {
        return sectionScores.reduce((total, section) => total + section.score, 0);
    }, [sectionScores]);

    // Get result based on total score
    const getResult = (score) => {
        for (const result of results) {
            if (score >= result.min && score <= result.max) {
                return result;
            }
        }
        return results[results.length - 1];
    };

    const hasInteracted = Object.values(checkedState).some(val => val);
    const result = getResult(totalScore);
    const displayResult = hasInteracted ? result : null;

    return (
        <main className="flex-grow flex flex-col items-center px-6 pb-32">
            {/* TITLE */}
            <h1 className="text-2xl font-bold mt-10 uppercase tracking-terminal text-center text-ink">
                [ COMPATIBILITY QUIZ ]
            </h1>

            <h2 className="text-center text-ink/60 mt-4 mb-10 leading-relaxed max-w-2xl text-sm uppercase tracking-terminal">
                Check all that apply. If unsure, answer no.
            </h2>

            {/* SECTIONS */}
            {sections.map((section, sectionIndex) => (
                <div key={section.id} className="w-full max-w-2xl mb-12">
                    {/* Section Header */}
                    <div className="flex justify-between items-center border-b border-structure pb-2 mb-4">
                        <h3 className="text-sm font-bold uppercase tracking-terminal text-ink">
                            [ SECTION {sectionIndex + 1}: {section.name.toUpperCase()} ]
                        </h3>
                        <span className="text-sm font-bold uppercase tracking-terminal text-ink">
                            {sectionScores[sectionIndex]?.score || 0} PTS
                        </span>
                    </div>

                    {/* Questions */}
                    <div className="flex flex-col gap-2">
                        {section.questions.map((q) => (
                            <div 
                                className={`flex items-start space-x-3 p-3 border border-structure hover:bg-highlight hover:text-invert hover:border-highlight transition-none cursor-pointer ${checkedState[q.id] ? 'bg-highlight text-invert border-highlight' : ''}`} 
                                key={q.id}
                                onClick={() => handleOnChange(q.id)}
                            >
                                <span className="text-sm font-semibold w-8 flex-shrink-0">
                                    {checkedState[q.id] ? '[*]' : '[ ]'}
                                </span>
                                <span className="text-sm cursor-pointer select-none flex-grow">
                                    {q.text}
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            ))}

            {/* STICKY RESULTS */}
            <div className="fixed bottom-0 left-0 right-0 bg-canvas border-t border-structure py-6 text-center">
                <div className="container mx-auto px-6">
                    <h3 className="text-2xl font-bold text-ink mb-2 uppercase tracking-terminal">
                        TOTAL SCORE: {totalScore}
                    </h3>
                    {displayResult ? (
                        <div>
                            <span className="text-lg font-bold uppercase tracking-terminal text-ink">
                                {displayResult.label}
                            </span>
                            <p className="text-xs text-ink/70 mt-2 max-w-xl mx-auto">
                                {displayResult.description}
                            </p>
                        </div>
                    ) : (
                        <span className="text-sm text-ink/70 uppercase tracking-terminal">
                            [ NOT ENOUGH DATA ]
                        </span>
                    )}
                </div>
            </div>
        </main>
    );
};

export default AreYouCompatible;
