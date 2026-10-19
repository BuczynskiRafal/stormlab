"""Reading the four summary tables of a SWMM report, absences included."""

from __future__ import annotations

from pathlib import Path

import pytest

from stormlab.io import rpt

EMPTY_SECTIONS_REPORT = Path("models/recomendations/dataset-backup/proba/generated") / (
    "01_recomendation_template_D1000_EulerRain60_S6_0_rural.rpt"
)
"""A tracked report SWMM produced with nothing surcharged and nothing flooded.

The two "no table here" sentences are pinned literals in the parser, so at least
one assertion has to hold them against a report the engine actually wrote rather
than against a string this test also made up.
"""


def _report(
    *,
    links: str = "  C1                   CONDUIT     0.500     0  00:17      1.20    0.35    0.40\n",
    surcharge: str | None = "  C1                          0.20      0.20      0.20      0.05         0.20\n",
    depths: str = "  J1                   JUNCTION     0.10     0.50   100.50     0  00:16        0.40\n",
    flooding: str | None = "  J1                      0.20     0.010      0  00:17       0.005     0.000\n",
    runoff: str = "-1.880",
    routing: str = "0.366",
) -> str:
    """A report with the shape SWMM writes, and only the rows a test cares about."""
    surcharge_body = (
        "  ----------------------------------------------------------------------------\n"
        "                                                           Hours        Hours \n"
        "                         --------- Hours Full --------   Above Full   Capacity\n"
        "  Conduit                Both Ends  Upstream  Dnstream   Normal Flow   Limited\n"
        "  ----------------------------------------------------------------------------\n" + surcharge
        if surcharge is not None
        else f"  {rpt.NO_CONDUITS_SURCHARGED}\n"
    )
    flooding_body = (
        "  --------------------------------------------------------------------------\n"
        "                                                             Total   Maximum\n"
        "                                 Maximum   Time of Max       Flood    Ponded\n"
        "                        Hours       Rate    Occurrence      Volume    Volume\n"
        "  Node                 Flooded       CMS   days hr:min    10^6 ltr   1000 m3\n"
        "  --------------------------------------------------------------------------\n" + flooding
        if flooding is not None
        else f"  {rpt.NO_NODES_FLOODED}\n"
    )
    return (
        "  *************\n"
        "  Runoff Quantity Continuity     hectare-m            mm\n"
        "  *************\n"
        f"  Continuity Error (%) .....     {runoff}\n"
        "\n"
        "  *************\n"
        "  Flow Routing Continuity        hectare-m      10^6 ltr\n"
        "  *************\n"
        f"  Continuity Error (%) .....     {routing}\n"
        "\n"
        "  ******************\n"
        "  Node Depth Summary\n"
        "  ******************\n"
        "\n"
        "  ---------------------------------------------------------------------------------\n"
        "                                 Average  Maximum  Maximum  Time of Max    Reported\n"
        "                                   Depth    Depth      HGL   Occurrence   Max Depth\n"
        "  Node                 Type       Meters   Meters   Meters  days hr:min      Meters\n"
        "  ---------------------------------------------------------------------------------\n"
        f"{depths}"
        "\n"
        "  *********************\n"
        "  Node Flooding Summary\n"
        "  *********************\n"
        "\n"
        f"{flooding_body}"
        "\n"
        "  ********************\n"
        "  Link Flow Summary\n"
        "  ********************\n"
        "\n"
        "  -----------------------------------------------------------------------------\n"
        "                                 Maximum  Time of Max   Maximum    Max/    Max/\n"
        "                                  |Flow|   Occurrence   |Veloc|    Full    Full\n"
        "  Link                 Type          CMS  days hr:min     m/sec    Flow   Depth\n"
        "  -----------------------------------------------------------------------------\n"
        f"{links}"
        "\n"
        "  *************************\n"
        "  Conduit Surcharge Summary\n"
        "  *************************\n"
        "\n"
        f"{surcharge_body}"
        "\n"
        "  Analysis begun on:  Mon Aug  4 06:13:55 2025\n"
        "  Analysis ended on:  Mon Aug  4 06:13:55 2025\n"
    )


class TestTheFourTables:
    def test_link_flow_keeps_the_four_measurements_of_every_conduit(self) -> None:
        frame = rpt.link_flow_summary(_report())

        assert list(frame.columns) == list(rpt.LINK_FLOW_COLUMNS)
        assert frame.loc["C1", "max_flow_cms"] == 0.5
        assert frame.loc["C1", "max_velocity_mps"] == 1.2
        assert frame.loc["C1", "max_full_flow_ratio"] == 0.35
        assert frame.loc["C1", "max_full_depth_ratio"] == 0.4

    def test_conduit_surcharge_keeps_the_five_hour_counts(self) -> None:
        frame = rpt.conduit_surcharge_summary(_report())

        assert list(frame.columns) == list(rpt.CONDUIT_SURCHARGE_COLUMNS)
        assert frame.loc["C1", "hours_above_normal_flow"] == 0.05
        assert frame.loc["C1", "hours_capacity_limited"] == 0.2

    def test_node_depth_keeps_the_maximum_the_hgl_and_the_reported_maximum(self) -> None:
        frame = rpt.node_depth_summary(_report())

        assert frame.loc["J1", "avg_depth_m"] == 0.1
        assert frame.loc["J1", "max_depth_m"] == 0.5
        assert frame.loc["J1", "max_hgl_m"] == 100.5
        assert frame.loc["J1", "max_depth_reported_m"] == 0.4

    def test_node_flooding_keeps_the_hours_and_the_volume(self) -> None:
        frame = rpt.node_flooding_summary(_report())

        assert frame.loc["J1", "hours_flooded"] == 0.2
        assert frame.loc["J1", "total_flood_volume_1e6l"] == 0.005

    def test_continuity_errors_keep_their_sign(self) -> None:
        assert rpt.continuity_errors(_report(runoff="-1.880", routing="0.366")) == (-1.88, 0.366)

    def test_a_non_conduit_link_is_skipped_rather_than_parsed_into_a_short_row(self) -> None:
        text = _report(
            links=(
                "  P1                   PUMP        0.100     0  00:17\n"
                "  C1                   CONDUIT     0.500     0  00:17      1.20    0.35    0.40\n"
            )
        )

        frame = rpt.link_flow_summary(text)

        assert list(frame.index) == ["C1"]


class TestAnAbsentTable:
    def test_no_conduits_surcharged_reads_as_an_empty_table_not_as_an_error(self) -> None:
        frame = rpt.conduit_surcharge_summary(_report(surcharge=None))

        assert frame.empty
        assert list(frame.columns) == list(rpt.CONDUIT_SURCHARGE_COLUMNS)

    def test_no_nodes_flooded_reads_as_an_empty_table_not_as_an_error(self) -> None:
        frame = rpt.node_flooding_summary(_report(flooding=None))

        assert frame.empty

    def test_an_element_absent_from_an_exception_table_carries_zero_hours(self) -> None:
        empty = rpt.conduit_surcharge_summary(_report(surcharge=None))

        filled = rpt.zero_filled(empty, ["C1", "C2"], rpt.CONDUIT_SURCHARGE_COLUMNS)

        assert filled.loc["C1", "hours_capacity_limited"] == 0.0
        assert not filled.isna().to_numpy().any()

    def test_a_listed_element_keeps_its_hours_while_the_others_get_zero(self) -> None:
        surcharged = rpt.conduit_surcharge_summary(_report())

        filled = rpt.zero_filled(surcharged, ["C1", "C2"], rpt.CONDUIT_SURCHARGE_COLUMNS)

        assert filled.loc["C1", "hours_capacity_limited"] == 0.2
        assert filled.loc["C2", "hours_capacity_limited"] == 0.0

    def test_a_node_absent_from_the_depth_table_is_not_reported_as_zero(self) -> None:
        frame = rpt.node_depth_summary(_report())

        reindexed = frame.reindex(["J1", "J2"])

        assert reindexed.loc["J1", "max_depth_m"] == 0.5
        assert reindexed.loc["J2"].isna().all()


class TestRefusals:
    def test_a_report_without_a_section_is_named_in_the_error(self) -> None:
        text = _report().replace("Link Flow Summary", "Link Flow Digest")

        with pytest.raises(rpt.RptParseError, match="Link Flow Summary"):
            rpt.link_flow_summary(text)

    def test_a_row_with_the_wrong_number_of_fields_is_refused_with_the_row(self) -> None:
        text = _report(depths="  J1                   JUNCTION     0.10     0.50\n")

        with pytest.raises(rpt.RptParseError, match="has 4 fields, expected 8"):
            rpt.node_depth_summary(text)

    def test_a_non_numeric_value_is_refused_rather_than_coerced(self) -> None:
        text = _report(links="  C1                   CONDUIT     0.500     0  00:17      n/a    0.35    0.40\n")

        with pytest.raises(rpt.RptParseError, match="not numeric"):
            rpt.link_flow_summary(text)

    def test_an_element_named_twice_is_refused(self) -> None:
        text = _report(
            links=(
                "  C1                   CONDUIT     0.500     0  00:17      1.20    0.35    0.40\n"
                "  C1                   CONDUIT     0.700     0  00:18      1.30    0.45    0.50\n"
            )
        )

        with pytest.raises(rpt.RptParseError, match="more than once"):
            rpt.link_flow_summary(text)

    def test_a_missing_report_file_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(rpt.RptParseError, match="does not exist"):
            rpt.read_report(tmp_path / "absent.rpt")

    def test_a_missing_continuity_section_is_refused(self) -> None:
        text = _report().replace("Flow Routing Continuity", "Flow Routing Digest")

        with pytest.raises(rpt.RptParseError, match="routing continuity"):
            rpt.continuity_errors(text)


class TestAgainstAReportTheEngineWrote:
    """The pinned sentences and the table shape, held against a tracked report."""

    def test_the_empty_section_sentences_are_the_ones_swmm_writes(self, repo_root_path: Path) -> None:
        report_path = repo_root_path / EMPTY_SECTIONS_REPORT
        if not report_path.is_file():
            pytest.skip(f"tracked dataset report not present in standalone checkout: {EMPTY_SECTIONS_REPORT}")
        text = report_path.read_text(encoding="utf-8", errors="replace")

        assert rpt.NO_CONDUITS_SURCHARGED in text
        assert rpt.NO_NODES_FLOODED in text

    def test_a_real_report_parses_into_two_populated_and_two_empty_tables(self, repo_root_path: Path) -> None:
        report_path = repo_root_path / EMPTY_SECTIONS_REPORT
        if not report_path.is_file():
            pytest.skip(f"tracked dataset report not present in standalone checkout: {EMPTY_SECTIONS_REPORT}")
        tables = rpt.read_report(report_path)

        assert len(tables.link_flow) == 11
        assert len(tables.node_depth) == 13
        assert tables.conduit_surcharge.empty
        assert tables.node_flooding.empty
        assert tables.routing_continuity_error_percent == 0.366
