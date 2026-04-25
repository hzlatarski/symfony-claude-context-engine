<?php
namespace App;

use App\Service\Helper;

class Caller
{
    public function takesParam(Helper $h): void
    {
        $h->doIt();
    }

    public function newExpr(): void
    {
        $h = new Helper();
        $h->doIt();
    }

    public function untyped($x): void
    {
        $x->doIt();
    }
}
