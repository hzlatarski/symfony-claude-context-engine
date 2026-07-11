<?php

namespace App;

// WHY: this class exists to demonstrate rationale extraction
class Rationale
{
    public function withInlineHack(): void
    {
        // HACK: force a repaint because the avatar iframe swallows the first frame
        $x = 1;
        # TODO: replace this with the new pipeline once it ships
        echo $x;
    }

    /**
     * @deprecated use withInlineHack() instead — kept for the 0.2 API
     */
    public function oldMethod(): void
    {
        echo 2;
    }

    /**
     * Legacy price calculator kept for the 0.1 billing API.
     *
     * Note that this once handled proration too.
     *
     * @deprecated use PriceService::calculate() — summary-first docblock
     * @see PriceService
     */
    public function summaryFirstDeprecated(): void
    {
        echo 4;
    }

    public function clean(): void
    {
        // Note that this is an ordinary comment, not a rationale tag
        echo 3;
    }
}
