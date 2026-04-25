<?php
namespace App;

class Child extends Base
{
    public function callsParent(): void
    {
        parent::inherited();
    }

    public function callsThis(): void
    {
        $this->inherited();
    }
}
