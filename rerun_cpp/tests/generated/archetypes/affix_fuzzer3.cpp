// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/cpp/mod.rs
// Based on "crates/re_types/definitions/rerun/testing/archetypes/fuzzy.fbs".

#include "affix_fuzzer3.hpp"

namespace rerun {
    namespace archetypes {
        const char AffixFuzzer3::INDICATOR_COMPONENT_NAME[] =
            "rerun.testing.components.AffixFuzzer3Indicator";
    }

    Result<std::vector<SerializedComponentBatch>> AsComponents<archetypes::AffixFuzzer3>::serialize(
        const archetypes::AffixFuzzer3& archetype
    ) {
        using namespace archetypes;
        std::vector<SerializedComponentBatch> cells;
        cells.reserve(18);

        if (archetype.fuzz2001.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer1>(archetype.fuzz2001.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2002.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer2>(archetype.fuzz2002.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2003.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer3>(archetype.fuzz2003.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2004.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer4>(archetype.fuzz2004.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2005.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer5>(archetype.fuzz2005.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2006.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer6>(archetype.fuzz2006.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2007.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer7>(archetype.fuzz2007.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2008.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer8>(archetype.fuzz2008.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2009.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer9>(archetype.fuzz2009.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2010.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer10>(archetype.fuzz2010.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2011.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer11>(archetype.fuzz2011.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2012.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer12>(archetype.fuzz2012.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2013.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer13>(archetype.fuzz2013.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2014.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer14>(archetype.fuzz2014.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2015.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer15>(archetype.fuzz2015.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2016.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer16>(archetype.fuzz2016.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2017.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer17>(archetype.fuzz2017.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        if (archetype.fuzz2018.has_value()) {
            auto result =
                ComponentBatch<rerun::components::AffixFuzzer18>(archetype.fuzz2018.value())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }
        {
            auto result =
                ComponentBatch<AffixFuzzer3::IndicatorComponent>(AffixFuzzer3::IndicatorComponent())
                    .serialize();
            RR_RETURN_NOT_OK(result.error);
            cells.emplace_back(std::move(result.value));
        }

        return cells;
    }
} // namespace rerun